#include "matmul_lockfree.h"
#include "kernel_utils.h"
#include "utils.h"

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

#define NC 5000
#define MC 256
#define KC 512

#define min(x, y) ((x) < (y) ? (x) : (y))
#define ceil_div(x, n) (((x) + (n) - 1) / (n))
#define round_up(x, n) (((x) + (n) - 1) / (n) * (n))

struct threadpool {
  atomic_bool stop;

  atomic_int counter;

  atomic_bool generation;

  atomic_int current_chunk_1;
  atomic_int current_chunk_2;

  int n_threads;

  struct thread_compute_state *workers;
  float *blockB_packed;

  pthread_mutex_t job_mutex;
  pthread_cond_t job_cond;

  unsigned long job_epoch;

  struct matmul_params *params;

  int completed_workers;

  pthread_mutex_t completion_mutex;
  pthread_cond_t completion_cond;
};

static void pack_panelB(float *B, float *blockB_packed, int nr, int kc, int K) {
  for (int p = 0; p < kc; p++) {

    for (int j = 0; j < nr; j++) {
      *blockB_packed++ = B[j * K + p];
    }

    for (int j = nr; j < NR; j++) {
      *blockB_packed++ = 0.0f;
    }
  }
}

static void pack_blockB(float *B, float *packed, int nc, int kc, int K) {
  for (int j = 0; j < nc; j += NR) {

    int nr = min(NR, nc - j);

    pack_panelB(&B[j * K], &packed[j * kc], nr, kc, K);
  }
}

static void pack_panelA(float *A, float *blockA_packed, int mr, int kc, int M) {
  for (int p = 0; p < kc; p++) {

    for (int i = 0; i < mr; i++) {
      *blockA_packed++ = A[p * M + i];
    }

    for (int i = mr; i < MR; i++) {
      *blockA_packed++ = 0.0f;
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

      __asm__ __volatile__("pause");
    }
  }
}

static void matmul_compute_one_chunk(struct matmul_params *params,
                                     struct thread_compute_state *state, int nc,
                                     int kc, int i, int j, int p, bool load) {
  struct threadpool *tpool = state->tpool;
  float *blockA_packed = state->blockA_packed;
  float *blockB_packed = tpool->blockB_packed;

  float *A = params->A;
  float *C = params->C;

  int M = params->M;

  int mc = min(MC, M - i);

  pack_blockA(&A[p * M + i], blockA_packed, mc, kc, M);

  for (int jr = 0; jr < nc; jr += NR) {

    int nr = min(NR, nc - jr);

    for (int ir = 0; ir < mc; ir += MR) {

      int mr = min(MR, mc - ir);

      kernel_16x6_accum(&blockA_packed[ir * kc], &blockB_packed[jr * kc],

                        &C[(j + jr) * M + (i + ir)],

                        mr, nr, kc, M, load);
    }
  }
}

static void matmul_execute(struct threadpool *tpool,
                           struct thread_compute_state *state) {
  struct matmul_params *params = tpool->params;

  float *A = params->A;
  float *B = params->B;

  int M = params->M;
  int N = params->N;
  int K = params->K;

  int ith = state->ith;

  for (int j = 0; j < N; j += NC) {

    int nc = min(NC, N - j);

    bool load = false;

    for (int p = 0; p < K; p += KC) {

      if (p != 0) {
        load = true;
      }

      int kc = min(KC, K - p);

      int n_chunks_1 = ceil_div(nc, NR);

      int n_chunks_2 = ceil_div(M, MC);

      int current_chunk_1 = ith;

      while (current_chunk_1 < n_chunks_1) {

        int jr = current_chunk_1 * NR;

        int nr = min(NR, nc - jr);

        pack_panelB(&B[(j + jr) * K + p],

                    &tpool->blockB_packed[jr * kc],

                    nr, kc, K);

        current_chunk_1 = atomic_fetch_add_explicit(&tpool->current_chunk_1, 1,
                                                    memory_order_relaxed);
      }

      threadpool_barrier(tpool);

      int current_chunk_2 = ith;

      while (current_chunk_2 < n_chunks_2) {

        int i = current_chunk_2 * MC;

        matmul_compute_one_chunk(params, state, nc, kc, i, j, p, load);

        current_chunk_2 = atomic_fetch_add_explicit(&tpool->current_chunk_2, 1,
                                                    memory_order_relaxed);
      }

      threadpool_barrier(tpool);

      if (ith == 0) {

        atomic_store_explicit(&tpool->current_chunk_1, tpool->n_threads,
                              memory_order_relaxed);

        atomic_store_explicit(&tpool->current_chunk_2, tpool->n_threads,
                              memory_order_relaxed);
      }

      threadpool_barrier(tpool);
    }
  }

  (void)A;
}

static void *threadpool_worker(void *data) {
  struct thread_compute_state *state = (struct thread_compute_state *)data;

  struct threadpool *tpool = state->tpool;

  unsigned long local_epoch = 0;

  for (;;) {

    pthread_mutex_lock(&tpool->job_mutex);

    while (!atomic_load_explicit(&tpool->stop, memory_order_acquire) &&
           tpool->job_epoch == local_epoch) {

      pthread_cond_wait(&tpool->job_cond, &tpool->job_mutex);
    }

    if (atomic_load_explicit(&tpool->stop, memory_order_acquire)) {

      pthread_mutex_unlock(&tpool->job_mutex);
      break;
    }

    local_epoch = tpool->job_epoch;

    pthread_mutex_unlock(&tpool->job_mutex);

    matmul_execute(tpool, state);

    pthread_mutex_lock(&tpool->completion_mutex);

    tpool->completed_workers++;

    if (tpool->completed_workers == tpool->n_threads) {

      pthread_cond_signal(&tpool->completion_cond);
    }

    pthread_mutex_unlock(&tpool->completion_mutex);
  }

  return NULL;
}

struct threadpool *threadpool_init(int num_threads) {
  if (num_threads <= 0) {
    return NULL;
  }

  struct threadpool *tpool = calloc(1, sizeof(*tpool));

  if (!tpool) {
    return NULL;
  }

  tpool->n_threads = num_threads;

  atomic_init(&tpool->stop, false);
  atomic_init(&tpool->counter, num_threads);
  atomic_init(&tpool->generation, false);

  atomic_init(&tpool->current_chunk_1, num_threads);
  atomic_init(&tpool->current_chunk_2, num_threads);

  tpool->job_epoch = 0;
  tpool->params = NULL;
  tpool->blockB_packed = aligned_malloc(round_up(NC, NR) * KC * sizeof(float));

  tpool->completed_workers = 0;

  if (!tpool->blockB_packed) {
    free(tpool);
    return NULL;
  }

  if (pthread_mutex_init(&tpool->job_mutex, NULL) != 0) {

    free(tpool);
    return NULL;
  }

  if (pthread_cond_init(&tpool->job_cond, NULL) != 0) {

    pthread_mutex_destroy(&tpool->job_mutex);
    free(tpool);
    return NULL;
  }

  if (pthread_mutex_init(&tpool->completion_mutex, NULL) != 0) {

    pthread_cond_destroy(&tpool->job_cond);
    pthread_mutex_destroy(&tpool->job_mutex);
    free(tpool);
    return NULL;
  }

  if (pthread_cond_init(&tpool->completion_cond, NULL) != 0) {

    pthread_mutex_destroy(&tpool->completion_mutex);
    pthread_cond_destroy(&tpool->job_cond);
    pthread_mutex_destroy(&tpool->job_mutex);
    free(tpool);
    return NULL;
  }

  tpool->workers = calloc(num_threads, sizeof(*tpool->workers));

  if (!tpool->workers) {

    pthread_cond_destroy(&tpool->completion_cond);
    pthread_mutex_destroy(&tpool->completion_mutex);

    pthread_cond_destroy(&tpool->job_cond);
    pthread_mutex_destroy(&tpool->job_mutex);

    free(tpool);
    return NULL;
  }

  for (int i = 0; i < num_threads; i++) {

    tpool->workers[i].tpool = tpool;
    tpool->workers[i].ith = i;
    tpool->workers[i].blockA_packed =
        aligned_malloc(round_up(MC, MR) * KC * sizeof(float));

    int rc = pthread_create(&tpool->workers[i].thread, NULL, threadpool_worker,
                            &tpool->workers[i]);

    if (!tpool->workers[i].blockA_packed || rc != 0) {

      atomic_store_explicit(&tpool->stop, true, memory_order_release);

      pthread_mutex_lock(&tpool->job_mutex);

      pthread_cond_broadcast(&tpool->job_cond);

      pthread_mutex_unlock(&tpool->job_mutex);

      for (int j = 0; j < i; j++) {
        pthread_join(tpool->workers[j].thread, NULL);
      }

      free(tpool->workers);

      pthread_cond_destroy(&tpool->completion_cond);

      pthread_mutex_destroy(&tpool->completion_mutex);

      pthread_cond_destroy(&tpool->job_cond);

      pthread_mutex_destroy(&tpool->job_mutex);

      free(tpool);

      return NULL;
    }
  }

  return tpool;
}

void threadpool_submit(struct threadpool *tpool, float *A, float *B, float *C,
                       int M, int N, int K) {
  struct matmul_params *params = malloc(sizeof(*params));

  if (!params) {
    abort();
  }

  params->A = A;
  params->B = B;
  params->C = C;

  params->M = M;
  params->N = N;
  params->K = K;

  pthread_mutex_lock(&tpool->job_mutex);

  if (tpool->params != NULL) {

    pthread_mutex_unlock(&tpool->job_mutex);

    free(params);

    abort();
  }

  tpool->params = params;

  atomic_store_explicit(&tpool->current_chunk_1, tpool->n_threads,
                        memory_order_relaxed);

  atomic_store_explicit(&tpool->current_chunk_2, tpool->n_threads,
                        memory_order_relaxed);

  atomic_store_explicit(&tpool->counter, tpool->n_threads,
                        memory_order_relaxed);

  tpool->job_epoch++;

  pthread_mutex_lock(&tpool->completion_mutex);

  tpool->completed_workers = 0;

  pthread_mutex_unlock(&tpool->completion_mutex);

  pthread_cond_broadcast(&tpool->job_cond);

  pthread_mutex_unlock(&tpool->job_mutex);
}

void threadpool_wait(struct threadpool *tpool) {
  pthread_mutex_lock(&tpool->completion_mutex);

  while (tpool->completed_workers < tpool->n_threads) {

    pthread_cond_wait(&tpool->completion_cond, &tpool->completion_mutex);
  }

  pthread_mutex_unlock(&tpool->completion_mutex);

  pthread_mutex_lock(&tpool->job_mutex);

  free(tpool->params);
  tpool->params = NULL;

  pthread_mutex_unlock(&tpool->job_mutex);
}

void threadpool_destroy(struct threadpool *tpool) {
  if (!tpool) {
    return;
  }

  if (tpool->params != NULL) {
    threadpool_wait(tpool);
  }

  pthread_mutex_lock(&tpool->job_mutex);

  atomic_store_explicit(&tpool->stop, true, memory_order_release);

  pthread_cond_broadcast(&tpool->job_cond);

  pthread_mutex_unlock(&tpool->job_mutex);

  for (int i = 0; i < tpool->n_threads; i++) {

    free(tpool->workers[i].blockA_packed);
    pthread_join(tpool->workers[i].thread, NULL);
  }

  pthread_cond_destroy(&tpool->completion_cond);

  pthread_mutex_destroy(&tpool->completion_mutex);

  pthread_cond_destroy(&tpool->job_cond);

  pthread_mutex_destroy(&tpool->job_mutex);

  free(tpool->workers);
  free(tpool);
}

void matmul_lockfree(float *A, float *B, float *C, int M, int N, int K) {
  struct threadpool *tpool = threadpool_init(NTHREADS);

  if (!tpool) {
    abort();
  }

  threadpool_submit(tpool, A, B, C, M, N, K);

  threadpool_wait(tpool);

  threadpool_destroy(tpool);
}
