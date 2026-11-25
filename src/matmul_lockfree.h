#pragma once

#include <pthread.h>
#include <stdatomic.h>
#include <stdbool.h>

struct matmul_params {
  float *A;
  float *B;
  float *C;

  int M;
  int N;
  int K;
};

struct threadpool;

struct thread_compute_state {
  struct threadpool *tpool;
  pthread_t thread;
  int ith;
  float *blockA_packed;
};

struct threadpool *threadpool_init(int num_threads);

void threadpool_submit(struct threadpool *tpool, float *A, float *B, float *C,
                       int M, int N, int K);

void threadpool_wait(struct threadpool *tpool);

void threadpool_destroy(struct threadpool *tpool);

void matmul_lockfree(float *A, float *B, float *C, int M, int N, int K);
