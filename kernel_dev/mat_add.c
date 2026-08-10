#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// void mat_add(const float *A, const float *B, float *C, int M, int N) {
//   for (size_t i = 0; i < M; ++i) {
//     for (size_t j = 0; j < N; ++j) {
//       C[i * N + j] = A[i * N + j] + B[i * N + j];
//     }
//   }
// }
inline void kernel_8x5(const float *A, const float *B, float *C, int M, int N) {
  __m256 C_accum[5] = {}; // zero-initialized
  __m256 b_pack[5];
  __m256 a_pack[5];
  a_pack[0] = _mm256_loadu_ps(&A[0]);
  b_pack[0] = _mm256_loadu_ps(&B[0]);
  C_accum[0] = _mm256_add_ps(a_pack[0], b_pack[0]);

  a_pack[1] = _mm256_loadu_ps(&A[M]);
  b_pack[1] = _mm256_loadu_ps(&B[M]);
  C_accum[1] = _mm256_add_ps(a_pack[1], b_pack[1]);

  a_pack[2] = _mm256_loadu_ps(&A[2 * M]);
  b_pack[2] = _mm256_loadu_ps(&B[2 * M]);
  C_accum[2] = _mm256_add_ps(a_pack[2], b_pack[2]);

  a_pack[3] = _mm256_loadu_ps(&A[3 * M]);
  b_pack[3] = _mm256_loadu_ps(&B[3 * M]);
  C_accum[3] = _mm256_add_ps(a_pack[3], b_pack[3]);

  a_pack[4] = _mm256_loadu_ps(&A[4 * M]);
  b_pack[4] = _mm256_loadu_ps(&B[4 * M]);
  C_accum[4] = _mm256_add_ps(a_pack[4], b_pack[4]);
  for (int i = 0; i < 5; ++i) {
    _mm256_storeu_ps(&C[i * M], C_accum[i]);
  }
}

// void mat_add(const float *A, const float *B, float *C, int M, int N) {
//   __m256 C_accum[5] = {}; // zero-initialized
//   __m256 b_pack[5];
//   __m256 a_pack[5];
//
//   for (int i = 0; i < M; i += 8) {
//     for (int j = 0; j < N; j += 5) {
//       // kernel_8x5(&A[j * M + i], &B[j * M + i], &C[j * M + i], M, N);
//       a_pack[0] = _mm256_loadu_ps(&A[j * M + i]);
//       b_pack[0] = _mm256_loadu_ps(&B[j * M + i]);
//       C_accum[0] = _mm256_add_ps(a_pack[0], b_pack[0]);
//
//       a_pack[1] = _mm256_loadu_ps(&A[j * M + i + M]);
//       b_pack[1] = _mm256_loadu_ps(&B[j * M + i + M]);
//       C_accum[1] = _mm256_add_ps(a_pack[1], b_pack[1]);
//
//       a_pack[2] = _mm256_loadu_ps(&A[j * M + i + 2 * M]);
//       b_pack[2] = _mm256_loadu_ps(&B[j * M + i + 2 * M]);
//       C_accum[2] = _mm256_add_ps(a_pack[2], b_pack[2]);
//
//       a_pack[3] = _mm256_loadu_ps(&A[j * M + i + 3 * M]);
//       b_pack[3] = _mm256_loadu_ps(&B[j * M + i + 3 * M]);
//       C_accum[3] = _mm256_add_ps(a_pack[3], b_pack[3]);
//
//       a_pack[4] = _mm256_loadu_ps(&A[j * M + i + 4 * M]);
//       b_pack[4] = _mm256_loadu_ps(&B[j * M + i + 4 * M]);
//       C_accum[4] = _mm256_add_ps(a_pack[4], b_pack[4]);
//       for (int k = 0; k < 5; ++k) {
//         _mm256_storeu_ps(&C[j * M + i + k * M], C_accum[k]);
//       }
//     }
//   }
// }
void mat_add(const float *A, const float *B, float *C, int M, int N) {
  for (int i = 0; i < M; ++i) {
    int j = 0;
    for (; j + 7 < N; j += 8) {
      __m256 a = _mm256_loadu_ps(&A[i * N + j]);
      __m256 b = _mm256_loadu_ps(&B[i * N + j]);
      __m256 c = _mm256_add_ps(a, b);
      _mm256_storeu_ps(&C[i * N + j], c);
    }

    for (; j < N; ++j)
      C[i * N + j] = A[i * N + j] + B[i * N + j];
  }
}

int main() {
  srand(0);

  const size_t M = 128 * 128;
  const size_t N = 128 * 125;
  float *A = malloc(M * N * sizeof(float));
  float *B = malloc(M * N * sizeof(float));
  float *C = malloc(M * N * sizeof(float));

  if (!A || !B || !C) {
    fprintf(stderr, "Memory allocation failed\n");
    free(A);
    free(B);
    free(C);
    return 1;
  }

  for (uint64_t i = 0; i < M * N; ++i) {
    A[i] = (float)rand() / (float)RAND_MAX;
  }
  for (uint64_t i = 0; i < M * N; ++i) {
    B[i] = (float)rand() / (float)RAND_MAX;
  }
  clock_t start_time = clock();

  // matmul_kernel1(A, B, C, M, N, K);
  // matmul_kernel1_row_major(A, B, C, M, N, K);
  mat_add(A, B, C, M, N);

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
