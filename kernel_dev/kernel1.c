#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
//
// // MxK x KxN = MxN
// void matmul_kernel1(float *A, float *B, float *C, int M, int N, int K) {
//   for (int n = 0; n < N; ++n) {
//     for (int m = 0; m < M; ++m) {
//       for (int k = 0; k < K; ++k) {
//         C[n * M + m] += A[k * M + m] * B[n * K + k];
//       }
//     }
//   }
// }
//
// int main() {
//   srand(0);
//   int M = 1024, N = 1024, K = 1024;
//   float A[1024 * 1024], B[1024 * 1024], C[1024 * 1024];
//   for (size_t i = 0; i < 1024 * 1024; ++i) {
//     A[i] = (float)rand();
//     B[i] = (float)rand();
//     C[i] = 0;
//   }
//   clock_t start_time = clock();
//   matmul_kernel1(A, B, C, M, N, K);
//   clock_t end_time = clock();
//   double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;
//
//   printf("Execution time: %f seconds\n", time_taken);
// }
//
//
// #include <stdio.h>
// #include <stdlib.h>
// #include <time.h>
//
// Computes C = A * B
// A: M x K (column-major)
// B: K x N (column-major)
// C: M x N (column-major)
// void matmul_kernel1(const float *A, const float *B, float *C, int M, int N,
//                     int K) {
//   for (int n = 0; n < N; ++n) {
//     for (int m = 0; m < M; ++m) {
//       float sum = 0.0f;
//       for (int k = 0; k < K; ++k) {
//         sum += A[k * M + m] * B[n * K + k];
//       }
//       C[n * M + m] = sum;
//     }
//   }
// }
void matmul_kernel1_col_major(const float *A, const float *B, float *C, int M,
                              int N, int K) {
  for (int n = 0; n < N; ++n) {

    // float acc = 0;
    for (int k = 0; k < K; ++k) {

      for (int m = 0; m < M; ++m) {
        C[n * M + m] += A[k * M + m] * B[n * K + k];
        // acc += A[k * M + m] * B[n * K + k];
      }
    }
    // C[n * M + m] = acc;
  }
}
void matmul_kernel1_row_major(const float *A, const float *B, float *C, int M,
                              int N, int K) {
  for (int m = 0; m < M; ++m) {

    // float acc = 0;
    for (int k = 0; k < K; ++k) {
      for (int n = 0; n < N; ++n) {

        C[m * N + n] += A[m * K + k] * B[k * N + n];
        // acc += A[k * M + m] * B[n * K + k];
      }
    }
    // C[n * M + m] = acc;
  }
}
void kernel_16x6(const float *A, const float *B, float *C, int M, int N,
                 int K) {
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

  for (int i = 0; i < N; ++i) {
    _mm256_storeu_ps(&C[i * M], C_accum[i][0]);
    _mm256_storeu_ps(&C[i * M + 8], C_accum[i][1]);
  }
}

void matmul_kernel1(const float *A, const float *B, float *C, int M, int N,
                    int K) {
  for (int m = 0; m < M; m += 16) {
    for (int n = 0; n < N; n += 6) {
      kernel_16x6(&A[m], &B[n * K], &C[n * M + m], M, N, K);
    }
  }
}

int main(void) {
  srand(0);

  const int M = 1024;
  const int N = 1020;
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

  // matmul_kernel1(A, B, C, M, N, K);
  // matmul_kernel1_row_major(A, B, C, M, N, K);
  matmul_kernel1_col_major(A, B, C, M, N, K);
  // matmul_kernel1(A, B, C, M, N, K);

  clock_t end_time = clock();

  double time_taken = (double)(end_time - start_time) / CLOCKS_PER_SEC;

  printf("Execution time: %.6f seconds\n", time_taken);

  // Prevent the compiler from optimizing away the computation.
  printf("C[0] = %f\n", C[0]);

  free(A);
  free(B);
  free(C);

  return 0;
}
