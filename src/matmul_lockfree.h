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
struct matmul_thread_params {

  int nc;
  int kc;

  int i;
  int j;
  int p;

  bool load;
};

struct thread_compute_state {

  struct threadpool *tpool;
  pthread_t thread;
  int ith;
  // struct matmul_thread_params *thrd_params;

  float *blockA_packed;
};

struct threadpool {
  // pthread_mutex_t mutex;
  // pthread_cond_t cond;

  atomic_bool stop;
  atomic_int counter;
  atomic_bool generation;
  atomic_int current_chunk_1;
  atomic_int current_chunk_2;

  int n_threads;
  struct thread_compute_state *workers;

  float *blockB_packed;
  struct matmul_params *params;
};

// static struct threadpool *threadpool_init(struct matmul_params *params,
//                                           int num_threads);
// static void threadpool_destroy(struct threadpool *tpool);
// static void *threadpool_worker(void *state);
// static void threadpool_barrier(struct threadpool *tpool);
//
// static void matmul_compute_one_chunk(struct matmul_params *params,
//                                      struct matmul_thread_params
//                                      *thrd_params);
void matmul_lockfree(float *A, float *B, float *C, int M, int N, int K);
