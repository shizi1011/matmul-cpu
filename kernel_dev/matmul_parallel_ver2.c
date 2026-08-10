#include "threadpool.h"
#include <immintrin.h>
// #include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define MR 16
#define NR 6

#define NTHREADS 8
#define MC MR * NTHREADS * 2  // 640
#define NC NR * NTHREADS * 20 // 2400
#define KC 500

// #define OMP_PRAGMA_PARALLEL _Pragma("omp parallel for num_threads(NTHREADS)")

#define min(x, y) ((x) < (y) ? (x) : (y))
#define ceil(x, n) (((x) + (n) - 1) / (n))

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

static void pack_panelB(float *B, float *blockB_packed, int nr, int kc, int K) {
  for (int p = 0; p < kc; p++) {
    for (int j = 0; j < nr; j++) {
      *blockB_packed++ = B[j * K + p];
    }
    for (int j = nr; j < NR; j++) {
      *blockB_packed++ = 0;
    }
  }
}

static void pack_blockB(float *B, float *blockB_packed, int nc, int kc, int K) {
  // #pragma omp parallel for
  for (int j = 0; j < nc; j += NR) {
    int nr = min(NR, nc - j);
    pack_panelB(&B[j * K], &blockB_packed[j * kc], nr, kc, K);
  }
}

static void pack_panelA(float *A, float *blockA_packed, int mr, int kc, int M) {
  for (int p = 0; p < kc; p++) {
    for (int i = 0; i < mr; i++) {
      *blockA_packed++ = A[p * M + i];
    }
    for (int i = mr; i < MR; i++) {
      *blockA_packed++ = 0;
    }
  }
}

static void pack_blockA(float *A, float *blockA_packed, int mc, int kc, int M) {
  for (int i = 0; i < mc; i += MR) {
    int mr = min(MR, mc - i);
    pack_panelA(&A[i], &blockA_packed[i * kc], mr, kc, M);
  }
}

static inline void maskload_accum(float *C, __m256 C_accum[6][2],
                                  __m256i packed_masks[2], int M, int nr) {
  for (int j = 0; j < nr; j++) {
    C_accum[j][0] = _mm256_maskload_ps(&C[j * M], packed_masks[0]);
    C_accum[j][1] = _mm256_maskload_ps(&C[j * M + 8], packed_masks[1]);
  }
}

static inline void load_accum(float *C, __m256 C_accum[6][2], int M, int nr) {
  for (int j = 0; j < nr; j++) {
    C_accum[j][0] = _mm256_loadu_ps(&C[j * M]);
    C_accum[j][1] = _mm256_loadu_ps(&C[j * M + 8]);
  }
}

static inline void maskstore_accum(float *C, __m256 C_accum[6][2],
                                   __m256i packed_masks[2], int M, int nr) {
  for (int j = 0; j < nr; j++) {
    _mm256_maskstore_ps(&C[j * M], packed_masks[0], C_accum[j][0]);
    _mm256_maskstore_ps(&C[j * M + 8], packed_masks[1], C_accum[j][1]);
  }
}

static inline void store_accum(float *C, __m256 C_accum[6][2], int M, int nr) {
  for (int j = 0; j < nr; j++) {
    _mm256_storeu_ps(&C[j * M], C_accum[j][0]);
    _mm256_storeu_ps(&C[j * M + 8], C_accum[j][1]);
  }
}

static inline void fma_loop(float *blockA_packed, float *blockB_packed,
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

static inline void build_masks(__m256i packed_masks[2], int mr) {
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

static void kernel_16x6_accum(float *blockA_packed, float *blockB_packed,
                              float *C, int mr, int nr, int kc, int M,
                              bool load) {

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
static void matmul_worker(void *thread_args) {

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
  int i = args->i;
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
  free(args);
}

void matmul_parallel(float *A, float *B, float *C, int M, int N, int K) {
  threadpool_t *threadpool = threadpool_create(NTHREADS);
  struct ThreadArgs base_args = {.A = A,
                                 .B = B,
                                 .C = C,

                                 .M = M,
                                 .N = N,
                                 .K = K,

                                 .j = 0,
                                 .p = 0,
                                 .nc = 0,
                                 .kc = 0,
                                 .load = false

  };

  for (int j = 0; j < N; j += NC) {
    int nc = min(NC, N - j);
    bool load = false;
    for (int p = 0; p < K; p += KC) {
      if (p != 0)
        load = true;

      int kc = min(KC, K - p);
      pack_blockB(&B[j * K + p], blockB_packed, nc, kc, K);

      int n_works = ceil(M, MC);
      for (size_t work = 0; work < n_works; ++work) {
        struct ThreadArgs *args = malloc(sizeof(struct ThreadArgs));
        *args = base_args;
        {
          args->j = j;
          args->p = p;
          args->nc = nc;
          args->kc = kc;
          args->load = load;
          args->i = work * MC;
        }
        threadpool_add_work(threadpool, matmul_worker, args);
      }
      threadpool_wait(threadpool);
    }
  }

  threadpool_destroy(threadpool);
}
