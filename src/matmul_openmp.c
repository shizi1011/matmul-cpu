#include "kernel_utils.h"
#include <immintrin.h>
#include <stdbool.h>
#include <stdint.h>

#define MR 16
#define NR 6

#ifndef NTHREADS
#define NTHREADS 8
#endif

#define MC MR * NTHREADS * 2  // 640
#define NC NR * NTHREADS * 20 // 2400
#define KC 500

#define min(x, y) ((x) < (y)) ? (x) : (y)

static float blockA_packed[MC * KC] __attribute__((aligned(64)));
#pragma omp threadprivate(blockA_packed)

static float blockB_packed[NC * KC] __attribute__((aligned(64)));

// static int8_t mask[32] __attribute__((aligned(64))) = {
//     -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
//     0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};

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
#pragma omp for
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

void matmul_openmp(float *A, float *B, float *C, int M, int N, int K) {
#pragma omp parallel num_threads(NTHREADS)
  for (int j = 0; j < N; j += NC) {
    int nc = min(NC, N - j);
    bool load = false;
    for (int p = 0; p < K; p += KC) {
      if (p != 0)
        load = true;
      int kc = min(KC, K - p);
      pack_blockB(&B[j * K + p], blockB_packed, nc, kc, K);
#pragma omp barrier

#pragma omp for
      for (int i = 0; i < M; i += MC) {
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
      }
    }
  }
}
