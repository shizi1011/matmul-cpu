#ifndef NTHREADS
#define NHHREADS 8
#endif

void matmul_naive(float *A, float *B, float *C, int m, int n, int k) {
#pragma omp parallel for collapse(2) num_threads(NTHREADS)
  for (int i = 0; i < m; i++) {
    for (int j = 0; j < n; j++) {
      float accumulator = 0;
      for (int p = 0; p < k; p++) {
        accumulator += A[p * m + i] * B[j * k + p];
      }
      C[j * m + i] = accumulator;
    }
  }
}
