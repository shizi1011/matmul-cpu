#include "matmul.h"
#include "utils.h"
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#include <cblas.h>

// #ifdef LOCKFREE
// #include "matmul_lockfree_v2.h"
// #endif

#ifdef GGML_MATMUL
void matmul_ggml(float *A, float *B, float *C, int M, int N, int K);
#endif

int main(int argc, char *argv[]) {
  srand(time(NULL));
  int m = 1000;
  int n = 1000;
  int k = 1000;

  if (argc > 3) {
    m = atoi(argv[1]);
    n = atoi(argv[2]);
    k = atoi(argv[3]);
  }
  printf("================\n");
  printf("M = %i\nN = %i\nK = %i\n", m, n, k);
  printf("================\n");

  int n_failed = 0;

  float *A = (float *)_mm_malloc(m * k * sizeof(float), 64);
  float *B = (float *)_mm_malloc(n * k * sizeof(float), 64);
  float *C = (float *)_mm_malloc(m * n * sizeof(float), 64);
  float *C_ref = (float *)_mm_malloc(m * n * sizeof(float), 64);

#ifdef GGML_MATMUL
  float *B_transposed = (float *)_mm_malloc(n * k * sizeof(float), 64);
  float *C_ref_transposed = (float *)_mm_malloc(m * n * sizeof(float), 64);

#endif

  init_rand(A, m * k);
  init_rand(B, k * n);

// #ifdef OPENBLAS
//     cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A, m,
//     B,
//                 k, 0, C, m);
// #else
//     matmul_openmp(A, B, C, m, n, k);
//
// #endif
// matmul_naive(A, B, C_ref, m, n, k);
#if defined(OPENBLAS)
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A, m, B, k,
              0, C, m);
#elif defined(OPENMP)
  matmul_openmp(A, B, C, m, n, k);

#elif defined(THREADPOOL)
  matmul_threadpool(A, B, C, m, n, k);

#elif defined(LOCKFREE)
  matmul_lockfree(A, B, C, m, n, k);

#elif defined(GGML_MATMUL)
  for (int j = 0; j < n; ++j) {
    for (int i = 0; i < k; ++i) {
      B_transposed[i * n + j] = B[j * k + i];
    }
  }

  matmul_ggml(A, B_transposed, C, m, n, k);
#else
  matmul_naive(A, B, C, m, n, k);
#endif

  // matmul_naive(A, B, C_ref, m, n, k);
  cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A, m, B, k,
              0, C_ref, m);
  // cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A, k, B,
  // n,
  //             0, C_ref, n);

#ifdef GGML_MATMUL

  for (int i = 0; i < m; ++i) {
    for (int j = 0; j < n; ++j) {
      C_ref_transposed[j * m + i] = C_ref[i * n + j];
    }
  }

  struct val_stat_t result = validate_mat(C, C_ref_transposed, m * n, 1e-4);
#else
  struct val_stat_t result = validate_mat(C, C_ref, m * n, 1e-4);

#endif

  if (result.n_error > 0) {
    printf("Test : FAILED\n");
  } else {
    printf("Test : PASSED\n");
  }
  _mm_free(A);
  _mm_free(B);
  _mm_free(C);
  _mm_free(C_ref);

#ifdef GGML_MATMUL
  _mm_free(B_transposed);
  _mm_free(C_ref_transposed);
#endif
  printf("=====================\n");
}
