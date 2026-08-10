#include "matmul_parallel_ver2.h"
#include <immintrin.h>
// #include <omp.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define MEMALIGN 64

void init_rand(float *mat, size_t n_elem) {
  for (size_t i = 0; i < n_elem; i++) {
    mat[i] = rand() / (float)RAND_MAX;
  }

  // #pragma omp parallel
  //   {
  //     // Thread-local seed initialized using the thread ID
  //     unsigned int seed = 1234 + omp_get_thread_num();
  //
  // #pragma omp for
  //     for (size_t i = 0; i < n_elem; i++) {
  //       mat[i] = (float)rand_r(&seed) / (float)RAND_MAX;
  // //     }
  //   }
}

uint64_t timer() {
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}

int main(void) {
  // srand(0);
  srand(time(NULL));

  int matsize = 5000;
  const int M = matsize;
  const int N = matsize;
  const int K = matsize;

  size_t sizeA = (size_t)M * K;
  size_t sizeB = (size_t)K * N;
  size_t sizeC = (size_t)M * N;

  // float *A = malloc(sizeA * sizeof(float));
  // float *B = malloc(sizeB * sizeof(float));
  // float *C = malloc(sizeC * sizeof(float));
  float *A = (float *)_mm_malloc(matsize * matsize * sizeof(float), MEMALIGN);
  float *B = (float *)_mm_malloc(matsize * matsize * sizeof(float), MEMALIGN);
  float *C = (float *)_mm_malloc(matsize * matsize * sizeof(float), MEMALIGN);

  if (!A || !B || !C) {
    fprintf(stderr, "Memory allocation failed\n");
    free(A);
    free(B);
    free(C);
    return 1;
  }
  init_rand(A, matsize * matsize);
  init_rand(B, matsize * matsize);

  // for (size_t i = 0; i < sizeA; ++i)
  //   A[i] = (float)rand() / (float)RAND_MAX;
  //
  // for (size_t i = 0; i < sizeB; ++i)
  //   B[i] = (float)rand() / (float)RAND_MAX;
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

  int n_iters = 3;
  uint64_t start = timer();

  for (int i = 0; i < n_iters; ++i) {

    // Get starting timestamp
    // timespec_get(&start, TIME_UTC);

    matmul_parallel(A, B, C, M, N, K);

    // Get ending timestamp
    // timespec_get(&end, TIME_UTC);

    // Calculate time differences in seconds and nanoseconds
  }
  uint64_t end = timer();

  double exec_time = (end - start) * 1e-9 / n_iters;

  double FLOP = 2 * (double)matsize * matsize * matsize;
  int gflops = (int)(FLOP / exec_time / 1e9);

  printf("Execution time: %.9f seconds, GFLOPS = %i\n", exec_time, gflops);

  // Prevent the compiler from optimizing away the computation.
  printf("C[0] = %f\n", C[0]);

  printf("C[-1] = %f\n", C[sizeC - 1]);

  _mm_free(A);
  _mm_free(B);
  _mm_free(C);

  // free(A);
  // free(B);
  // free(C);

  return 0;
}
