#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define min(x, y) ((x) < (y)) ? x : y

#define MR 16
#define NR 6

#define MC MR * 40  // 640
#define NC NR * 200 // 1200
#define KC 500

// static float blockA_buffer[MR * KC] __attribute__((aligned(64)));
// static float blockB_buffer[NR * KC] __attribute__((aligned(64)));

static float blockB_packed[KC * NC] __attribute__((aligned(64)));
static float blockA_packed[KC * MC] __attribute__((aligned(64)));

void kernel_16x6_pad(const float *A, const float *B, float *C, int M, int mc,
                     int nc, int mr, int nr, int kc, int blockA_1d) {

  __m256i masks[2];
  __m256 C_accum[6][2] = {}; // zero-initialized
  __m256 b_packFloat8;
  __m256 a0_packFloat8;
  __m256 a1_packFloat8;
  for (int i = 0; i < kc; ++i) {
    a0_packFloat8 = _mm256_loadu_ps(&A[i * blockA_1d]);
    a1_packFloat8 = _mm256_loadu_ps(&A[i * blockA_1d + 8]);

    b_packFloat8 = _mm256_broadcast_ss(&B[i]);
    C_accum[0][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[0][0]);
    C_accum[0][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[0][1]);

    b_packFloat8 = _mm256_broadcast_ss(&B[kc + i]);
    C_accum[1][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[1][0]);
    C_accum[1][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[1][1]);

    b_packFloat8 = _mm256_broadcast_ss(&B[2 * kc + i]);
    C_accum[2][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[2][0]);
    C_accum[2][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[2][1]);

    b_packFloat8 = _mm256_broadcast_ss(&B[3 * kc + i]);
    C_accum[3][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[3][0]);
    C_accum[3][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[3][1]);

    b_packFloat8 = _mm256_broadcast_ss(&B[4 * kc + i]);
    C_accum[4][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[4][0]);
    C_accum[4][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[4][1]);

    b_packFloat8 = _mm256_broadcast_ss(&B[5 * kc + i]);
    C_accum[5][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[5][0]);
    C_accum[5][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[5][1]);
  }

  if (mr != 16) {
    const uint32_t bit_mask = 65535; // 00000000 00000000 11111111 11111111
    masks[0] = _mm256_setr_epi32(bit_mask << (mr + 15), bit_mask << (mr + 14),
                                 bit_mask << (mr + 13), bit_mask << (mr + 12),
                                 bit_mask << (mr + 11), bit_mask << (mr + 10),
                                 bit_mask << (mr + 9), bit_mask << (mr + 8));
    masks[1] = _mm256_setr_epi32(bit_mask << (mr + 7), bit_mask << (mr + 6),
                                 bit_mask << (mr + 5), bit_mask << (mr + 4),
                                 bit_mask << (mr + 3), bit_mask << (mr + 2),
                                 bit_mask << (mr + 1), bit_mask << mr);
    for (int j = 0; j < nr; j++) {
      // _mm256_maskstore_ps(&C[j * M], masks[0], C_accum[j][0]);
      // _mm256_maskstore_ps(&C[j * M + 8], masks[1], C_accum[j][1]);
      __m256 c0 = _mm256_maskload_ps(&C[j * M], masks[0]);
      __m256 c1 = _mm256_maskload_ps(&C[j * M + 8], masks[1]);

      c0 = _mm256_add_ps(c0, C_accum[j][0]);
      c1 = _mm256_add_ps(c1, C_accum[j][1]);

      _mm256_maskstore_ps(&C[j * M], masks[0], c0);
      _mm256_maskstore_ps(&C[j * M + 8], masks[1], c1);
    }
  } else {
    for (int j = 0; j < nr; ++j) {
      // _mm256_storeu_ps(&C[i * M], C_accum[i][0]);
      // _mm256_storeu_ps(&C[i * M + 8], C_accum[i][1]);
      __m256 c0 = _mm256_loadu_ps(&C[j * M]);
      __m256 c1 = _mm256_loadu_ps(&C[j * M + 8]);

      c0 = _mm256_add_ps(c0, C_accum[j][0]);
      c1 = _mm256_add_ps(c1, C_accum[j][1]);

      _mm256_storeu_ps(&C[j * M], c0);
      _mm256_storeu_ps(&C[j * M + 8], c1);
    }
  }
}

void pack_panelB(const float *B, float *blockB_packed, int nr, int kc, int K) {

  for (int p = 0; p < kc; ++p) {
    for (int i = 0; i < nr; ++i) {
      *blockB_packed++ = B[i * K + p];
    }
    for (int i = nr; i < NR; ++i) {
      *blockB_packed++ = 0.0;
    }
  }
}

void pack_blockB(const float *B, float *blockB_packed, int nc, int kc, int K) {
  // blockB_packed = kc x nc
  for (int i = 0; i < nc; i += NR) {
    int nr = min(nr, nc - i);
    pack_panelB(&B[i * K], &blockB_packed[i * kc], nr, kc, K);
  }
}

void pack_panelA(const float *A, float *blockA_packed, int mr, int kc, int M) {
  for (int p = 0; p < kc; ++p) {
    for (int i = 0; i < mr; ++i) {
      *blockA_packed++ = A[i + p * M];
    }
    for (int i = mr; i < MR; ++i) {
      *blockA_packed++ = 0.0;
    }
  }
}

void pack_blockA(const float *A, float *blockA_packed, int mc, int kc, int M) {
  for (int i = 0; i < mc; i += MR) {
    int mr = min(MR, mc - i);
    pack_panelA(&A[i], &blockA_packed[i * kc], mr, kc, M);
  }
}

void pad_blockA(float *blockA_packed, float *blockA_buffer, int mr, int mc,
                int kc) {
  for (int k = 0; k < kc; ++k) {
    for (int i = 0; i < mr; ++i) {
      // MR x KC
      blockA_buffer[k * MR + i] = blockA_packed[k * MC + i];
    }
    for (int i = mr; i < MR; ++i) {
      blockA_buffer[k * MR + i] = 0.0;
    }
  }
}
void pad_blockB(float *blockB_packed, float *blockB_buffer, int kc, int nc,
                int nr) {
  for (int i = 0; i < NR; ++i) {
    for (int k = 0; k < kc; ++k) {
      // KC x NR
      if (i < nr)
        blockB_buffer[i * KC + k] = blockB_packed[i * KC + k];
      else
        blockB_buffer[i * KC + k] = 0.0;
    }
  }
}

void matmul_cache(float *A, float *B, float *C, const int M, const int N,
                  const int K) {
  for (int j = 0; j < N; j += NC) {

    const int nc = min(NC, N - j);

    for (int p = 0; p < K; p += KC) {

      const int kc = min(KC, K - p);
      pack_blockB(&B[j * K + p], blockB_packed, nc, kc, K);

      for (int i = 0; i < M; i += MC) {

        const int mc = min(MC, M - i);
        pack_blockA(&A[p * M + i], blockA_packed, mc, kc, M);
        for (int ir = 0; ir < mc; ir += MR) {

          const int mr = min(MR, mc - ir);
          float *blockA = &blockA_packed[ir];
          int blockA_1d = MC;

          if (mr != MR) {
            pad_blockA(blockA, blockA_buffer, mr, mc, kc);
            blockA = blockA_buffer;
            blockA_1d = MR;
            // printf("pad_blockA\n");
          }

          for (int jr = 0; jr < nc; jr += NR) {

            const int nr = min(NR, nc - jr);
            float *blockB = &blockB_packed[jr * KC];
            if (nr != NR) {
              pad_blockB(blockB, blockB_buffer, kc, nc, nr);
              blockB = blockB_buffer;
              // printf("pad_blockB\n");
            }

            // blockA_packed = mcxkc
            // blockB_packed = kcxnc

            // printf("mc=%d kc=%d ir=%d\n", mc, kc, ir);
            // printf("blockA offset = %td\n", blockA - blockA_packed);

            kernel_16x6_pad(blockA, blockB, &C[(j + jr) * M + (i + ir)], M, mc,
                            nc, mr, nr, kc, blockA_1d);
          }
        }
      }
    }
  }
}

int main(void) {
  srand(0);

  const int M = 1025;
  const int N = 1024;
  const int K = 1024;

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

  clock_t start_time = clock();

  matmul_cache(A, B, C, M, N, K);

  clock_t end_time = clock();

  double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;

  printf("Execution time: %.6f seconds\n", time_taken);

  // Prevent the compiler from optimizing away the computation.
  printf("C[-1] = %f\n", C[sizeC - 1]);

  free(A);
  free(B);
  free(C);

  return 0;
}
