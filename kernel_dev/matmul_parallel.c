#include <immintrin.h>
#include <pthread.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>

#define MR 16
#define NR 6

#define NTHREADS 8
#define MC MR * NTHREADS * 5  // 640
#define NC NR * NTHREADS * 50 // 2400
#define KC 500

#define OMP_PRAGMA_PARALLEL _Pragma("omp parallel for num_threads(NTHREADS)")

#define min(x, y) ((x) < (y) ? (x) : (y))
#define CEIL(x, n) (((x) + (n) - 1) / (n))

// static float blockA_packed[MC * KC] __attribute__((aligned(64)));
static float blockB_packed[NC * KC] __attribute__((aligned(64)));
static int8_t mask[32] __attribute__((aligned(64))) = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};

struct ThreadArgs {
  float *A;
  float *B;
  float *C;

  int M;
  int N;
  int K;

  int nc;
  int kc;

  int i;
  int j;
  int p;

  bool load;

  // int row_begin;
  // int row_end;
  //
  // int col_begin;
  // int col_end;
};

void pack_panelB(float *B, float *blockB_packed, int nr, int kc, int K) {
  for (int p = 0; p < kc; p++) {
    for (int j = 0; j < nr; j++) {
      *blockB_packed++ = B[j * K + p];
    }
    for (int j = nr; j < NR; j++) {
      *blockB_packed++ = 0;
    }
  }
}

void pack_blockB(float *B, float *blockB_packed, int nc, int kc, int K) {
  for (int j = 0; j < nc; j += NR) {
    int nr = min(NR, nc - j);
    pack_panelB(&B[j * K], &blockB_packed[j * kc], nr, kc, K);
  }
}

void pack_panelA(float *A, float *blockA_packed, int mr, int kc, int M) {
  for (int p = 0; p < kc; p++) {
    for (int i = 0; i < mr; i++) {
      *blockA_packed++ = A[p * M + i];
    }
    for (int i = mr; i < MR; i++) {
      *blockA_packed++ = 0;
    }
  }
}

void pack_blockA(float *A, float *blockA_packed, int mc, int kc, int M) {
  for (int i = 0; i < mc; i += MR) {
    int mr = min(MR, mc - i);
    pack_panelA(&A[i], &blockA_packed[i * kc], mr, kc, M);
  }
}

inline void maskload_accum(float *C, __m256 C_accum[6][2],
                           __m256i packed_masks[2], int M, int nr) {
  for (int j = 0; j < nr; j++) {
    C_accum[j][0] = _mm256_maskload_ps(&C[j * M], packed_masks[0]);
    C_accum[j][1] = _mm256_maskload_ps(&C[j * M + 8], packed_masks[1]);
  }
}

inline void load_accum(float *C, __m256 C_accum[6][2], int M, int nr) {
  for (int j = 0; j < nr; j++) {
    C_accum[j][0] = _mm256_loadu_ps(&C[j * M]);
    C_accum[j][1] = _mm256_loadu_ps(&C[j * M + 8]);
  }
}

inline void maskstore_accum(float *C, __m256 C_accum[6][2],
                            __m256i packed_masks[2], int M, int nr) {
  for (int j = 0; j < nr; j++) {
    _mm256_maskstore_ps(&C[j * M], packed_masks[0], C_accum[j][0]);
    _mm256_maskstore_ps(&C[j * M + 8], packed_masks[1], C_accum[j][1]);
  }
}

inline void store_accum(float *C, __m256 C_accum[6][2], int M, int nr) {
  for (int j = 0; j < nr; j++) {
    _mm256_storeu_ps(&C[j * M], C_accum[j][0]);
    _mm256_storeu_ps(&C[j * M + 8], C_accum[j][1]);
  }
}

inline void fma_loop(float *blockA_packed, float *blockB_packed,
                     __m256 C_accum[6][2], __m256 a0_packFloat8,
                     __m256 a1_packFloat8, __m256 b_packFloat8, int kc) {

  for (int p = 0; p < kc; p++) {
    a0_packFloat8 = _mm256_loadu_ps(blockA_packed);
    a1_packFloat8 = _mm256_loadu_ps(blockA_packed + 8);

    b_packFloat8 = _mm256_broadcast_ss(blockB_packed);
    C_accum[0][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[0][0]);
    C_accum[0][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[0][1]);

    b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 1);
    C_accum[1][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[1][0]);
    C_accum[1][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[1][1]);

    b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 2);
    C_accum[2][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[2][0]);
    C_accum[2][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[2][1]);

    b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 3);
    C_accum[3][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[3][0]);
    C_accum[3][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[3][1]);

    b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 4);
    C_accum[4][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[4][0]);
    C_accum[4][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[4][1]);

    b_packFloat8 = _mm256_broadcast_ss(blockB_packed + 5);
    C_accum[5][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[5][0]);
    C_accum[5][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[5][1]);

    blockA_packed += 16;
    blockB_packed += 6;
  }
}

inline void build_masks(__m256i packed_masks[2], int mr) {
  const uint32_t bit_mask = 65535;
  packed_masks[0] = _mm256_setr_epi32(
      bit_mask << (mr + 15), bit_mask << (mr + 14), bit_mask << (mr + 13),
      bit_mask << (mr + 12), bit_mask << (mr + 11), bit_mask << (mr + 10),
      bit_mask << (mr + 9), bit_mask << (mr + 8));
  packed_masks[1] = _mm256_setr_epi32(
      bit_mask << (mr + 7), bit_mask << (mr + 6), bit_mask << (mr + 5),
      bit_mask << (mr + 4), bit_mask << (mr + 3), bit_mask << (mr + 2),
      bit_mask << (mr + 1), bit_mask << mr);
}

void kernel_16x6_accum(float *blockA_packed, float *blockB_packed, float *C,
                       int mr, int nr, int kc, int M, bool load) {

  __m256 C_accum[6][2] = {};
  __m256 b_packFloat8 = {};
  __m256 a0_packFloat8 = {};
  __m256 a1_packFloat8 = {};

  if (mr != 16) {
    __m256i packed_masks[2];
    build_masks(packed_masks, mr);
    if (load)
      maskload_accum(C, C_accum, packed_masks, M, nr);
    fma_loop(blockA_packed, blockB_packed, C_accum, a0_packFloat8,
             a1_packFloat8, b_packFloat8, kc);
    maskstore_accum(C, C_accum, packed_masks, M, nr);
  } else {
    if (load)
      load_accum(C, C_accum, M, nr);
    fma_loop(blockA_packed, blockB_packed, C_accum, a0_packFloat8,
             a1_packFloat8, b_packFloat8, kc);
    store_accum(C, C_accum, M, nr);
  }
}

void *matmul_worker(void *thread_args) {

  float blockA_packed[MC * KC] __attribute__((aligned(64)));

  struct ThreadArgs *args = (struct ThreadArgs *)thread_args;
  float *A = args->A;
  float *B = args->B;
  float *C = args->C;

  int M = args->M;
  int N = args->N;
  int K = args->K;

  int nc = args->nc;
  int kc = args->kc;
  int i = args->i; // thread id
  int j = args->j;
  int p = args->p;
  bool load = args->load;

  int mc = min(MC, M - i);
  pack_blockA(&A[p * M + i], blockA_packed, mc, kc, M);
  for (int jr = 0; jr < nc; jr += NR) {
    int nr = min(NR, nc - jr);
    for (int ir = 0; ir < mc; ir += MR) {
      int mr = min(MR, mc - ir);
      kernel_16x6_accum(&blockA_packed[ir * kc], &blockB_packed[jr * kc],
                        &C[(j + jr) * M + (i + ir)], mr, nr, kc, M, load);
    }
  }
  return NULL;
}

void matmul_parallel(float *A, float *B, float *C, int M, int N, int K) {

  for (int j = 0; j < N; j += NC) {
    int nc = min(NC, N - j);
    bool load = false;
    for (int p = 0; p < K; p += KC) {
      if (p != 0)
        load = true;

      int kc = min(KC, K - p);
      pack_blockB(&B[j * K + p], blockB_packed, nc, kc, K);

      int num_steps = CEIL(M, MC * NTHREADS);
      for (int step = 0; step < num_steps; ++step) {
        int n_threads = min(NTHREADS, CEIL(M - MC * NTHREADS * step, MC));
        pthread_t threads[n_threads];
        struct ThreadArgs thread_args[n_threads];
        for (int i = 0; i < n_threads; ++i) {
          thread_args[i].A = A;
          thread_args[i].B = B;
          thread_args[i].C = C;

          thread_args[i].M = M;
          thread_args[i].N = N;
          thread_args[i].K = K;

          thread_args[i].nc = nc;
          thread_args[i].kc = kc;
          thread_args[i].i = step * MC * NTHREADS + i * MC;
          thread_args[i].j = j;
          thread_args[i].p = p;
          thread_args[i].load = load;
          pthread_create(&threads[i], NULL, matmul_worker, &thread_args[i]);
        }
        for (int i = 0; i < n_threads; ++i) {
          pthread_join(threads[i], NULL);
        }
      }

      // pthread_t threads[NTHREADS];
      // struct ThreadArgs thread_args[NTHREADS];
      // for (int i = 0; i < NTHREADS; ++i) {
      //   thread_args[i].A = A;
      //   thread_args[i].B = B;
      //   thread_args[i].C = C;
      //
      //   thread_args[i].M = M;
      //   thread_args[i].N = N;
      //   thread_args[i].K = K;
      //
      //   thread_args[i].nc = nc;
      //   thread_args[i].kc = kc;
      //   thread_args[i].i = i * MC;
      //   thread_args[i].j = j;
      //   thread_args[i].p = p;
      //   thread_args[i].load = load;
      //   pthread_create(&threads[i], NULL, matmul_worker, &thread_args[i]);
      // }
      // for (int i = 0; i < NTHREADS; ++i) {
      //   pthread_join(threads[i], NULL);
      // }
    }
  }
}

int main(void) {
  srand(0);

  const int M = 6000;
  const int N = 10000;
  const int K = 5000;

  size_t sizeA = (size_t)M * K;
  size_t sizeB = (size_t)K * N;
  size_t sizeC = (size_t)M * N;

  float *A = malloc(sizeA * sizeof(float));
  float *B = malloc(sizeB * sizeof(float));
  float *C = malloc(sizeC * sizeof(float));

  if (!A || !B || !C) {
    fprintf(stderr, "Memory allocation failed\n");
    free(A);
    free(B);
    free(C);
    return 1;
  }

  for (size_t i = 0; i < sizeA; ++i)
    A[i] = (float)rand() / (float)RAND_MAX;

  for (size_t i = 0; i < sizeB; ++i)
    B[i] = (float)rand() / (float)RAND_MAX;
  // for (size_t m = 0; m < M; ++m) {
  //   for (size_t k = 0; k < K; ++k) {
  //
  //     A[m * K + k] = (float)rand() / (float)RAND_MAX;
  //   }
  // }
  // for (size_t k = 0; k < K; ++k) {
  //
  //   for (size_t n = 0; n < N; ++n) {
  //
  //     B[k * N + n] = (float)rand() / (float)RAND_MAX;
  //   }
  // }
  // for (size_t k = 0; k < K; ++k) {
  //
  //   for (size_t m = 0; m < M; ++m) {
  //
  //     A[k * M + m] = (float)rand() / (float)RAND_MAX;
  //   }
  // }
  // for (size_t n = 0; n < N; ++n) {
  //
  //   for (size_t k = 0; k < K; ++k) {
  //
  //     B[n * K + k] = (float)rand() / (float)RAND_MAX;
  //   }
  // }

  struct timespec start, end;

  // Get starting timestamp
  timespec_get(&start, TIME_UTC);

  matmul_parallel(A, B, C, M, N, K);

  // Get ending timestamp
  timespec_get(&end, TIME_UTC);

  // Calculate time differences in seconds and nanoseconds
  double time_taken = (end.tv_sec - start.tv_sec) +
                      (double)(end.tv_nsec - start.tv_nsec) / 1000000000.0;

  printf("Execution time: %.9f seconds\n", time_taken);

  // Prevent the compiler from optimizing away the computation.
  printf("C[0] = %f\n", C[0]);

  printf("C[-1] = %f\n", C[sizeC - 1]);

  free(A);
  free(B);
  free(C);

  return 0;
}
