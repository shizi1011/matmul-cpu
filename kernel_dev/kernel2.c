#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#define min(x, y) ((x) < (y)) ? x : y

void kernel_16x6_pad(const float *A, const float *B, float *C, int M, int N,
                     int m, int n, int K) {

  __m256i masks[2];
  __m256 C_accum[6][2] = {}; // zero-initialized
  __m256 b_packFloat8;
  __m256 a0_packFloat8;
  __m256 a1_packFloat8;
  for (int i = 0; i < K; ++i) {
    a0_packFloat8 = _mm256_loadu_ps(&A[i * M]);
    a1_packFloat8 = _mm256_loadu_ps(&A[i * M + 8]);

    b_packFloat8 = _mm256_broadcast_ss(&B[i]);
    C_accum[0][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[0][0]);
    C_accum[0][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[0][1]);

    b_packFloat8 = _mm256_broadcast_ss(&B[K + i]);
    C_accum[1][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[1][0]);
    C_accum[1][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[1][1]);

    b_packFloat8 = _mm256_broadcast_ss(&B[2 * K + i]);
    C_accum[2][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[2][0]);
    C_accum[2][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[2][1]);

    b_packFloat8 = _mm256_broadcast_ss(&B[3 * K + i]);
    C_accum[3][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[3][0]);
    C_accum[3][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[3][1]);

    b_packFloat8 = _mm256_broadcast_ss(&B[4 * K + i]);
    C_accum[4][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[4][0]);
    C_accum[4][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[4][1]);

    b_packFloat8 = _mm256_broadcast_ss(&B[5 * K + i]);
    C_accum[5][0] = _mm256_fmadd_ps(a0_packFloat8, b_packFloat8, C_accum[5][0]);
    C_accum[5][1] = _mm256_fmadd_ps(a1_packFloat8, b_packFloat8, C_accum[5][1]);
  }

  if (m != 16) {
    const uint32_t bit_mask = 65535; // 00000000 00000000 11111111 11111111
    masks[0] = _mm256_setr_epi32(bit_mask << (m + 15), bit_mask << (m + 14),
                                 bit_mask << (m + 13), bit_mask << (m + 12),
                                 bit_mask << (m + 11), bit_mask << (m + 10),
                                 bit_mask << (m + 9), bit_mask << (m + 8));
    masks[1] = _mm256_setr_epi32(bit_mask << (m + 7), bit_mask << (m + 6),
                                 bit_mask << (m + 5), bit_mask << (m + 4),
                                 bit_mask << (m + 3), bit_mask << (m + 2),
                                 bit_mask << (m + 1), bit_mask << m);
    for (int j = 0; j < n; j++) {
      _mm256_maskstore_ps(&C[j * M], masks[0], C_accum[j][0]);
      _mm256_maskstore_ps(&C[j * M + 8], masks[1], C_accum[j][1]);
    }
  } else {
    for (int i = 0; i < n; ++i) {
      _mm256_storeu_ps(&C[i * M], C_accum[i][0]);
      _mm256_storeu_ps(&C[i * M + 8], C_accum[i][1]);
    }
  }
}

void matmul_kernel(const float *A, const float *B, float *C, int M, int N,
                   int K) {
  for (int i = 0; i < M; i += 16) {
    int m = min(16, M - i);

    for (int j = 0; j < N; j += 6) {
      int n = min(6, N - j);
      kernel_16x6_pad(&A[i], &B[j * K], &C[j * M + i], M, N, m, n, K);
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

  matmul_kernel(A, B, C, M, N, K);

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
