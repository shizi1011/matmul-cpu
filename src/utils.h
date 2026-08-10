#pragma once
// #include <math.h>
// #include <omp.h>
#include <stdint.h>
#include <stdio.h>
// #include <stdlib.h>

//
// #ifndef NTHREADS
// #define NTHREADS 8
// #endif
//
// #ifndef OMP_SCHEDULE
// #define OMP_SCHEDULE auto
// #endif
//
// #define PRAGMA_OMP_PARALLEL_FOR \
//   _Pragma("omp parallel for schedule(OMP_SCHEDULE) num_threads(NTHREADS)")

int compare_floats(const void *a, const void *b);
int get_niter(int matsize, int niter_start, int niter_end, int matsize_start,
              int matsize_end);
void init_rand(float *mat, size_t n_elem);
void init_const(float *mat, float value, size_t n_elem);
struct val_stat_t {
  int n_error;
  int n_nans;
  int n_inf;
};

struct val_stat_t validate_mat(float *mat, float *mat_ref, size_t n_elem,
                               float eps);
uint64_t timer();
void printfn(const char *str, int n);
