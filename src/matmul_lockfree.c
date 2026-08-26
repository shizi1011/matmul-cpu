#include "matmul_lockfree.h"
#include "kernel_utils.h"
#include <immintrin.h>
#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>
#include <stdlib.h>

#define MR 16
#define NR 6

#ifndef NTHREADS
#define NTHREADS 8
#endif

#define MC MR * NTHREADS * 2
#define NC NR * NTHREADS * 20
#define KC 500

#define min(x, y) ((x) < (y) ? (x) : (y))
#define ceil(x, n) (((x) + (n) - 1) / (n))

// static float blockA_packed[MC * KC] __attribute__((aligned(64)));
static float blockB_packed[NC * KC] __attribute__((aligned(64)));

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

static void threadpool_barrier(struct threadpool *tpool) {
  bool local_gen =
      atomic_load_explicit(&tpool->generation, memory_order_relaxed);

  if (atomic_fetch_sub_explicit(&tpool->counter, 1, memory_order_acq_rel) ==
      1) {
    atomic_store_explicit(&tpool->counter, tpool->n_threads,
                          memory_order_relaxed);

    atomic_store_explicit(&tpool->generation, !local_gen, memory_order_release);
  } else {
    while (atomic_load_explicit(&tpool->generation, memory_order_acquire) ==
           local_gen) {
      // _mm_pause();

      __asm__ __volatile__("pause");
    }
  }
}

static void matmul_compute_one_chunk(struct matmul_params *params,
                                     struct matmul_thread_params *thrd_params) {

  float blockA_packed[MC * KC] __attribute__((aligned(64)));

  float *A = params->A;
  // float *B = args->B;
  float *C = params->C;

  int M = params->M;
  // int N = args->N;
  // int K = args->K;

  int nc = thrd_params->nc;
  int kc = thrd_params->kc;
  int i = thrd_params->i;
  int j = thrd_params->j;
  int p = thrd_params->p;
  bool load = thrd_params->load;

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

static void *threadpool_worker(void *data) {
  struct thread_compute_state *state = (struct thread_compute_state *)data;
  struct threadpool *tpool = state->tpool;
  struct matmul_params *params = tpool->params;

  int ith = state->ith;
  float *A = params->A;
  float *B = params->B;
  float *C = params->C;

  int M = params->M;
  int N = params->N;
  int K = params->K;
  // int nc = nc;
  // int kc = kc;
  // int i = i;
  // int j = j;
  // int p = p;
  // bool load = load;

  // int n_chunks_1 = ceil(NC, NR);
  int n_chunks_1;

  int n_chunks_2 = ceil(M, MC);
  int current_chunk_1, current_chunk_2;

  // if (ith == 0) {
  //   // Every thread starts at ith, so the first unprocessed chunk is nth.
  //   This
  //   // save a bit of coordination right at the start.
  //   atomic_store_explicit(&tpool->current_chunk, tpool->n_threads,
  //                         memory_order_relaxed);
  // }

  for (int j = 0; j < N; j += NC) {
    int nc = min(NC, N - j);

    n_chunks_1 = ceil(nc, NR);

    bool load = false;
    for (int p = 0; p < K; p += KC) {
      if (p != 0)
        load = true;

      int kc = min(KC, K - p);

      // if (ith == 0) {
      //   pack_blockB(&B[j * K + p], blockB_packed, nc, kc, K);
      //
      //   atomic_store_explicit(&tpool->current_chunk, tpool->n_threads,
      //                         memory_order_relaxed);
      // }

      current_chunk_1 = ith;
      current_chunk_2 = ith;

      if (ith == 0) {
        atomic_store_explicit(&tpool->current_chunk_1, tpool->n_threads,
                              memory_order_relaxed);
        atomic_store_explicit(&tpool->current_chunk_2, tpool->n_threads,
                              memory_order_relaxed);
      }

      threadpool_barrier(tpool);

      while (current_chunk_1 < n_chunks_1) {

        int nr = min(NR, nc - current_chunk_1 * NR);

        pack_panelB(&B[current_chunk_1 * NR * K],
                    &blockB_packed[current_chunk_1 * NR * kc], nr, kc, K);

        current_chunk_1 = atomic_fetch_add_explicit(&tpool->current_chunk_1, 1,
                                                    memory_order_relaxed);
      }

      threadpool_barrier(tpool);

      while (current_chunk_2 < n_chunks_2) {
        struct matmul_thread_params thrd_params = {nc, kc, current_chunk_2 * MC,
                                                   j,  p,  load};

        matmul_compute_one_chunk(params, &thrd_params);

        current_chunk_2 = atomic_fetch_add_explicit(&tpool->current_chunk_2, 1,
                                                    memory_order_relaxed);
      }
      threadpool_barrier(tpool);
    }
  }
  return NULL;
}

static struct threadpool *threadpool_init(struct matmul_params *params,
                                          int num_threads) {

  struct threadpool *tpool = malloc(sizeof(struct threadpool));
  {
    tpool->n_threads = num_threads;
    tpool->params = params;

    tpool->stop = false;
    tpool->counter = num_threads;
    tpool->generation = false;
    tpool->current_chunk_1 = 0;
    tpool->current_chunk_2 = 0;
  }
  // pthread_mutex_init(&tpool->mutex, NULL);
  // pthread_cond_init(&tpool->cond, NULL);

  struct thread_compute_state *workers =
      calloc(num_threads, sizeof(struct thread_compute_state));

  for (int i = 0; i < num_threads; ++i) {
    workers[i].ith = i;
    workers[i].tpool = tpool;
  }
  tpool->workers = workers;

  for (int i = 0; i < num_threads; ++i) {
    pthread_create(&tpool->workers[i].thread, NULL, threadpool_worker,
                   &workers[i]);
  }

  return tpool;
}

static void threadpool_destroy(struct threadpool *tpool) {

  // pthread_mutex_lock(&tpool->mutex);
  //
  // tpool->stop = true;
  // pthread_cond_broadcast(&tpool->cond);

  for (int i = 0; i < tpool->n_threads; ++i) {
    pthread_join(tpool->workers[i].thread, NULL);
  }

  // pthread_mutex_unlock(&tpool->mutex);
  //
  // pthread_mutex_destroy(&tpool->mutex);
  // pthread_cond_destroy(&tpool->cond);
  free(tpool->workers);
  free(tpool);
}

void matmul_lockfree(float *A, float *B, float *C, int M, int N, int K) {

  // struct matmul_params params = {A, B, C, M, N, K};
  struct matmul_params *params = malloc(sizeof(struct matmul_params));
  {
    params->A = A;
    params->B = B;
    params->C = C;
    params->M = M;
    params->N = N;
    params->K = K;
  }

  struct threadpool *tpool = threadpool_init(params, NTHREADS);
  // threadpool_barrier(tpool);
  threadpool_destroy(tpool);
  free(params);
}
