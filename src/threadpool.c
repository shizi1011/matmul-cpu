#include "threadpool.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#define MAX_THREADS 8

struct threadpool_work {
  thread_func_t func;
  void *args;
};
typedef struct threadpool_work threadpool_work_t;

struct threadpool {
  threadpool_work_t *queue;
  size_t queue_capacity;
  size_t head;
  size_t tail;
  size_t task_cnt;

  pthread_mutex_t mutex;
  pthread_cond_t work_cond;    // there is work to be processed
  pthread_cond_t working_cond; // threre are no threads procssing

  pthread_t *workers;

  size_t working_cnt; // how many threads are actively processing work
  size_t n_threads;
  bool stop; // stop threads
};

static void *threadpool_worker(void *arg) {
  threadpool_t *threadpool = arg;
  threadpool_work_t work;

  while (1) {
    pthread_mutex_lock(&threadpool->mutex);

    while (threadpool->task_cnt == 0 && !threadpool->stop) {
      pthread_cond_wait(&threadpool->work_cond, &threadpool->mutex);
    }

    if (threadpool->stop && threadpool->task_cnt == 0) {
      pthread_mutex_unlock(&(threadpool->mutex));
      break;
    }

    work = threadpool->queue[threadpool->head];
    threadpool->head = (threadpool->head + 1) % threadpool->queue_capacity;
    threadpool->working_cnt++;
    threadpool->task_cnt--;

    pthread_mutex_unlock(&threadpool->mutex);

    work.func(work.args);

    pthread_mutex_lock(&threadpool->mutex);
    threadpool->working_cnt--;

    if (threadpool->working_cnt == 0 && threadpool->task_cnt == 0)
      pthread_cond_broadcast(&(threadpool->working_cond));
    pthread_mutex_unlock(&(threadpool->mutex));
  }

  return NULL;
}

threadpool_t *threadpool_create(size_t n_threads, size_t queue_capacity) {

  if (n_threads == 0 || n_threads > MAX_THREADS || queue_capacity == 0)
    return NULL;

  threadpool_t *threadpool = malloc(sizeof(threadpool_t));
  if (!threadpool)
    return NULL;

  {
    threadpool->n_threads = n_threads;
    threadpool->queue_capacity = queue_capacity;
    threadpool->head = 0;
    threadpool->tail = 0;
    threadpool->task_cnt = 0;
    threadpool->working_cnt = 0;
    threadpool->stop = false;
  }
  threadpool->queue = malloc(sizeof(threadpool_work_t) * queue_capacity);
  if (threadpool->queue == NULL) {
    free(threadpool);
    return NULL;
  }
  threadpool->workers = malloc(sizeof(pthread_t) * n_threads);
  memset(threadpool->workers, 0, sizeof(pthread_t) * n_threads);

  pthread_mutex_init(&(threadpool->mutex), NULL);
  pthread_cond_init(&(threadpool->work_cond), NULL);
  pthread_cond_init(&(threadpool->working_cond), NULL);

  for (size_t i = 0; i < n_threads; i++) {
    pthread_create(&threadpool->workers[i], NULL, threadpool_worker,
                   threadpool);
  }

  return threadpool;
}

void threadpool_destroy(threadpool_t *threadpool) {

  pthread_mutex_lock(&(threadpool->mutex));

  threadpool->stop = true;

  pthread_cond_broadcast(&(threadpool->work_cond));

  pthread_mutex_unlock(&(threadpool->mutex));

  for (size_t i = 0; i < threadpool->n_threads; ++i) {
    pthread_join(threadpool->workers[i], NULL);
  }

  pthread_mutex_destroy(&(threadpool->mutex));
  pthread_cond_destroy(&(threadpool->work_cond));
  pthread_cond_destroy(&(threadpool->working_cond));

  free(threadpool->queue);
  free(threadpool->workers);
  free(threadpool);
}

void threadpool_wait(threadpool_t *threadpool) {

  pthread_mutex_lock(&(threadpool->mutex));

  while (threadpool->task_cnt != 0 || threadpool->working_cnt != 0) {
    pthread_cond_wait(&(threadpool->working_cond), &(threadpool->mutex));
  }

  pthread_mutex_unlock(&(threadpool->mutex));
}

bool threadpool_add_work(threadpool_t *threadpool, thread_func_t func,
                         void *args) {

  pthread_mutex_lock(&threadpool->mutex);

  if (threadpool->stop || threadpool->task_cnt == threadpool->queue_capacity) {
    pthread_mutex_unlock(&threadpool->mutex);
    return false;
  }
  threadpool->queue[threadpool->tail].func = func;
  threadpool->queue[threadpool->tail].args = args;
  threadpool->tail = (threadpool->tail + 1) % threadpool->queue_capacity;
  threadpool->task_cnt++;

  pthread_cond_signal(&(threadpool->work_cond));

  pthread_mutex_unlock(&(threadpool->mutex));

  return true;
}
