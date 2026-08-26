#include "matmul.h"
#include "utils.h"
#include <assert.h>
#include <immintrin.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <sys/stat.h>
#include <time.h>

#ifdef OPENBLAS
#include <cblas.h>
#endif

#define MEMALIGN 64
#define max(x, y) ((x) > (y) ? (x) : (y))

int main(int argc, char *argv[]) {
  srand(time(NULL));
  int minsize = 200;
  int stepsize = 200;
  int npts = 40;
  int warmup_niter = 5;
  int niter_start = 1001;
  int niter_end = 5;
  char *save_dir = "benchmark_results";

  if (argc > 7) {
    minsize = atoi(argv[1]);
    stepsize = atoi(argv[2]);
    npts = atoi(argv[3]);
    warmup_niter = atoi(argv[4]);
    niter_start = atoi(argv[5]);
    niter_end = atoi(argv[6]);
    save_dir = argv[7];
  }

  assert(warmup_niter >= 0 && npts > 0 && minsize > 0 && stepsize > 0 &&
         (niter_start >= niter_end));

  int divider_len = 30;
  // Warm-up
  if (warmup_niter > 0) {
    int warmup_matsize = minsize + (int)(npts / 2 - 1) * stepsize;
    int m = warmup_matsize;
    int n = warmup_matsize;
    int k = warmup_matsize;
    float *A = (float *)_mm_malloc(
        warmup_matsize * warmup_matsize * sizeof(float), MEMALIGN);
    float *B = (float *)_mm_malloc(
        warmup_matsize * warmup_matsize * sizeof(float), MEMALIGN);
    float *C = (float *)_mm_malloc(
        warmup_matsize * warmup_matsize * sizeof(float), MEMALIGN);

    const char *warmup_title = "Warm-up";
    int warmup_title_padding = divider_len / 2 + strlen(warmup_title) / 2;

    printfn("=", divider_len);
    printf("\n");
    printf("%*s\n", warmup_title_padding, warmup_title);
    printfn("=", divider_len);
    printf("\n");

    for (int j = 0; j < warmup_niter; j++) {
      fflush(stdout);
      printf("\rm=n=k=%i: %i / %i", warmup_matsize, j + 1, warmup_niter);
      // matmul_parallel(A, B, C, m, n, k);
    }

    printf("\n\n");
    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
  }

  // Benchmark
  int *gflops_all = (int *)malloc(npts * sizeof(int));
  int *matsizes = (int *)malloc(npts * sizeof(int));
  for (int i = 0; i < npts; i++) {
    matsizes[i] = minsize + i * stepsize;
  }

  printfn("=", divider_len);
  printf("\n");
  const char *bench_title = "Benchmark: GEMM";
  int bench_title_padding = divider_len / 2 + strlen(bench_title) / 2;
  printf("%*s\n", bench_title_padding, bench_title);
  printfn("=", divider_len);
  printf("\n");

  for (int i = 0; i < npts; i++) {
    int matsize = matsizes[i];
    int m = matsize;
    int n = matsize;
    int k = matsize;
    // uint64_t start = timer();

    float *A = (float *)_mm_malloc(matsize * matsize * sizeof(float), MEMALIGN);
    float *B = (float *)_mm_malloc(matsize * matsize * sizeof(float), MEMALIGN);
    float *C = (float *)_mm_malloc(matsize * matsize * sizeof(float), MEMALIGN);

    // uint64_t end = timer();
    // double exec_time = (end - start) * 1e-9;
    // printf("m=n=k=%i, __mm_malloc time : %lf\n", matsize, exec_time);
    //
    // start = timer();

    init_rand(A, matsize * matsize);
    init_rand(B, matsize * matsize);

    // end = timer();
    // exec_time = (end - start) * 1e-9;
    // printf("m=n=k=%i, __init_rand time : %lf\n", matsize, exec_time);

    int n_iter = max(1, get_niter(matsize, niter_start, niter_end, minsize,
                                  matsizes[npts - 1]));

    // printf("matsize : %i, n_iter : %d\n", matsize, n_iter);
    // printf("n_iters : %d\n", n_iter);
    // int n_iter = 3;

    uint64_t start = timer();
    for (int j = 0; j < n_iter; j++) {
// #ifdef OPENBLAS
//       cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A,
//       m,
//                   B, k, 0, C, m);
// #else
//
//       matmul_threadpool(A, B, C, m, n, k);
//       // matmul_parallel(A, B, C, m, n, k);
// #endif
#if defined(OPENBLAS)
      cblas_sgemm(CblasColMajor, CblasNoTrans, CblasNoTrans, m, n, k, 1, A, m,
                  B, k, 0, C, m);
#elif defined(OPENMP)
      matmul_openmp(A, B, C, m, n, k);
#elif defined(THREADPOOL)
      matmul_threadpool(A, B, C, m, n, k);
#elif defined(LOCKFREE)
      matmul_lockfree(A, B, C, m, n, k);
#else
      matmul_naive(A, B, C, m, n, k);
#endif
    }
    uint64_t end = timer();

    double exec_time = (end - start) * 1e-9 / n_iter;
    double FLOP = 2 * (double)matsize * matsize * matsize;
    int gflops = (int)(FLOP / exec_time / 1e9);
    gflops_all[i] = gflops;

    printf("exec_time : %lf\n", exec_time);
    printf("m=n=k=%i | GFLOPS = %i\n", matsize, gflops);
    start = timer();

    _mm_free(A);
    _mm_free(B);
    _mm_free(C);
    // end = timer();
    // exec_time = (end - start) * 1e-9;
    // printf("m=n=k=%i, __mm_free time : %lf\n", matsize, exec_time);
  }

  printf("\n================\n");

  struct stat buffer;
  if (stat(save_dir, &buffer) == -1) {
    int status = mkdir(save_dir, 0700);
    if (status != 0) {
      printf("Error creating directory %s\n", save_dir);
      return -1;
    }
  }

  char *filename = "GEMM_threadpool.txt";
  char *save_path = malloc(strlen(save_dir) + strlen(filename) + 2);
  strcpy(save_path, save_dir);
  strcat(save_path, "/");
  strcat(save_path, filename);

  FILE *fptr;
  fptr = fopen(save_path, "w");
  if (fptr == NULL) {
    printf("Error opening file %s\n", filename);
    return -1;
  }
  for (int i = 0; i < npts; i++) {
    fprintf(fptr, "%i %i\n", matsizes[i], gflops_all[i]);
  }
  fclose(fptr);
  printf("Saved in %s\n", save_path);

  free(gflops_all);
  free(matsizes);
  return 0;
}
