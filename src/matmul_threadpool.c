#include "kernel_utils.h"
#include "threadpool.h"
#include <immintrin.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <time.h>

#define MR 16
#define NR 6

#ifndef NTHREADS
#define NTHREADS 8
#endif

#define MC MR * NTHREADS * 2  // 640
#define NC NR * NTHREADS * 20 // 2400
#define KC 500

#define min(x, y) ((x) < (y) ? (x) : (y))
#define ceil(x, n) (((x) + (n) - 1) / (n))

// static float blockA_packed[MC * KC] __attribute__((aligned(64)));
static float blockB_packed[NC * KC] __attribute__((aligned(64)));

struct ThreadArgs {
  float *A;
  // float *B;
  float *C;

  int M;
  // int N;
  // int K;

  int nc;
  int kc;

  int i;
  int j;
  int p;

  bool load;
};

static void pack_panelB(float *B, float *blockB_packed, int nr, int kc, int K) {
  for (int p = 0; p < kc; p++) {
    for (int j = 0; j < nr; j++) {
      *blockB_packed++ = B[j * K + p];
    }
    for (int j = nr; j < NR; j++) {
      *blockB_packed++ = 0;
    }
  }
}

static void pack_blockB(float *B, float *blockB_packed, int nc, int kc, int K) {

  // #pragma omp parallel for
  for (int j = 0; j < nc; j += NR) {
    int nr = min(NR, nc - j);
    pack_panelB(&B[j * K], &blockB_packed[j * kc], nr, kc, K);
  }
}

static void pack_panelA(float *A, float *blockA_packed, int mr, int kc, int M) {
  for (int p = 0; p < kc; p++) {
    for (int i = 0; i < mr; i++) {
      *blockA_packed++ = A[p * M + i];
    }
    for (int i = mr; i < MR; i++) {
      *blockA_packed++ = 0;
    }
  }
}

static void pack_blockA(float *A, float *blockA_packed, int mc, int kc, int M) {
  for (int i = 0; i < mc; i += MR) {
    int mr = min(MR, mc - i);
    pack_panelA(&A[i], &blockA_packed[i * kc], mr, kc, M);
  }
}

static void matmul_worker(void *thread_args) {

  float blockA_packed[MC * KC] __attribute__((aligned(64)));

  struct ThreadArgs *args = (struct ThreadArgs *)thread_args;
  float *A = args->A;
  // float *B = args->B;
  float *C = args->C;

  int M = args->M;
  // int N = args->N;
  // int K = args->K;

  int nc = args->nc;
  int kc = args->kc;
  int i = args->i;
  int j = args->j;
  int p = args->p;
  bool load = args->load;

  int mc = min(MC, M - i);
  pack_blockA(&A[p * M + i], blockA_packed, mc, kc, M);
  for (int jr = 0; jr < nc; jr += NR) {
    int nr = min(NR, nc - jr);
    for (int ir = 0; ir < mc; ir += MR) {
      int mr = min(MR, mc - ir);
      kernel_16x6_accum(&blockA_packed[ir * kc], &blockB_packed[jr * kc],
                        &C[(j + jr) * M + (i + ir)], mr, nr, kc, M, load);
    }
  }
  free(args);
}

void matmul_threadpool(float *A, float *B, float *C, int M, int N, int K) {
  threadpool_t *threadpool = threadpool_create(NTHREADS, 2048);
  struct ThreadArgs base_args = {.A = A,
                                 // .B = B,
                                 .C = C,

                                 .M = M,
                                 // .N = N,
                                 // .K = K,

                                 .j = 0,
                                 .p = 0,
                                 .nc = 0,
                                 .kc = 0,
                                 .load = false

  };

  for (int j = 0; j < N; j += NC) {
    int nc = min(NC, N - j);
    bool load = false;
    for (int p = 0; p < K; p += KC) {
      if (p != 0)
        load = true;

      int kc = min(KC, K - p);
      pack_blockB(&B[j * K + p], blockB_packed, nc, kc, K);

      int n_works = ceil(M, MC);
      for (size_t work = 0; work < n_works; ++work) {
        struct ThreadArgs *args = malloc(sizeof(struct ThreadArgs));
        *args = base_args;
        {
          args->j = j;
          args->p = p;
          args->nc = nc;
          args->kc = kc;
          args->load = load;
          args->i = work * MC;
        }
        threadpool_add_work(threadpool, matmul_worker, args);
      }
      threadpool_wait(threadpool);
    }
  }
  threadpool_destroy(threadpool);
}
