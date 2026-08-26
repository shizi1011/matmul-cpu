#include "utils.h"
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

#ifndef NTHREADS
#define NTHREADS 8
#endif

int get_niter(int matsize, int niter_start, int niter_end, int matsize_start,
              int matsize_end) {

  if (matsize_end == matsize_start || niter_start == niter_end)
    return niter_start;
  float a = ((float)niter_end - niter_start) * (matsize_start * matsize_end) /
            (matsize_start - matsize_end);
  float b = niter_start - a / matsize_start;
  return round(a / matsize + b);
}

// void init_rand(float *mat, size_t n_elem) {
//
// #pragma omp parallel
//   {
//     // Thread-local seed initialized using the thread ID
//     unsigned int seed = 1234 + omp_get_thread_num();
//
// #pragma omp for
//     for (size_t i = 0; i < n_elem; i++) {
//       mat[i] = (float)rand_r(&seed) / (float)RAND_MAX;
//     }
//   }
// }

void init_rand(float *mat, size_t n_elem) {
  for (size_t i = 0; i < n_elem; i++) {
    mat[i] = rand() / (float)RAND_MAX;
  }
}

struct val_stat_t validate_mat(float *mat, float *mat_ref, size_t n_elem,
                               float eps) {
  struct val_stat_t result = {0, 0, 0};
  for (size_t i = 0; i < n_elem; i++) {
    float value = mat[i];
    float value_ref = mat_ref[i];
    if (isnan(value)) {
      result.n_nans += 1;
      result.n_error += 1;
      continue;
    }
    if (isinf(value)) {
      result.n_inf += 1;
      result.n_error += 1;
      continue;
    }
    if (fabsf((value - value_ref) / value_ref) > eps) {
      result.n_error += 1;
      continue;
    }
  }
  return result;
}

uint64_t timer() {
  struct timespec start;
  clock_gettime(CLOCK_MONOTONIC_RAW, &start);
  return (uint64_t)start.tv_sec * 1000000000 + (uint64_t)start.tv_nsec;
}

void printfn(const char *str, int n) {
  for (int i = 0; i < n; i++) {
    printf("%s", str);
  }
}

void *aligned_malloc(size_t size) {
  const int alignment = 64;

  if (size == 0) {
    return NULL;
  }
  void *aligned_memory = NULL;
  int result = posix_memalign(&aligned_memory, alignment, size);

  if (result != 0)
    return NULL;

  return aligned_memory;
}
