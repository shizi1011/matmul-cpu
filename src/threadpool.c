#include "threadpool.h"
#include <pthread.h>
#include <stdlib.h>
#include <string.h>

#define MAX_THREADS 8

struct threadpool_work {
  thread_func_t func;
  void *args;
  struct threadpool_work *next;
};
typedef struct threadpool_work threadpool_work_t;

struct threadpool {
  threadpool_work_t *work_begin;
  threadpool_work_t *work_end;

  pthread_mutex_t mutex;
  pthread_cond_t work_cond;    // there is work to be processed
  pthread_cond_t working_cond; // threre are no threads procssing

  pthread_t *workers;

  size_t working_cnt; // how many threads are actively processing work
  size_t n_threads;
  bool stop; // stop threads
};

static threadpool_work_t *threadpool_work_create(thread_func_t func,
                                                 void *args) {
  threadpool_work_t *work = malloc(sizeof(threadpool_work_t));
  if (!work) // malloc failed to allocate memory
    return NULL;
  work->func = func;
  work->args = args;
  work->next = NULL;
  return work;
}

static void threadpool_work_destroy(threadpool_work_t *work) {
  if (!work)
    return;
  free(work);
}

static threadpool_work_t *threadpool_work_get(threadpool_t *threadpool) {
  threadpool_work_t *work = threadpool->work_begin;
  if (!work)
    return NULL;
  if (!work->next) {
    threadpool->work_begin = threadpool->work_end = NULL;
  } else {
    threadpool->work_begin = work->next;
  }
  return work;
}

static void *threadpool_worker(void *arg) {
  threadpool_t *threadpool = arg;
  threadpool_work_t *work;

  while (1) {
    pthread_mutex_lock(&threadpool->mutex);
    while (!threadpool->work_begin && !threadpool->stop) {
      pthread_cond_wait(&threadpool->work_cond, &threadpool->mutex);
    }

    if (threadpool->stop && !threadpool->work_begin)
      break;

    work = threadpool_work_get(threadpool);
    threadpool->working_cnt++;
    pthread_mutex_unlock(&threadpool->mutex);

    if (work) {
      work->func(work->args);
      threadpool_work_destroy(work);
    }

    pthread_mutex_lock(&threadpool->mutex);
    threadpool->working_cnt--;
    if (!threadpool->stop && threadpool->working_cnt == 0 &&
        !threadpool->work_begin)
      pthread_cond_signal(&(threadpool->working_cond));
    pthread_mutex_unlock(&(threadpool->mutex));
  }

  pthread_mutex_unlock(&(threadpool->mutex));
  return NULL;
}

threadpool_t *threadpool_create(size_t n_threads) {

  threadpool_t *threadpool = malloc(sizeof(threadpool_t));
  {
    threadpool->n_threads = n_threads;
    threadpool->work_begin = NULL;
    threadpool->work_end = NULL;
    threadpool->working_cnt = 0;
    threadpool->stop = false;
  }

  pthread_mutex_init(&(threadpool->mutex), NULL);
  pthread_cond_init(&(threadpool->work_cond), NULL);
  pthread_cond_init(&(threadpool->working_cond), NULL);

  // pthread_t *workers = malloc(sizeof(pthread_t) * n_threads);
  // memset(workers, 0, sizeof(pthread_t) * n_threads);
  pthread_t *workers = calloc(n_threads, sizeof(pthread_t));
  threadpool->workers = workers;

  for (size_t i = 0; i < n_threads; i++) {
    pthread_create(&workers[i], NULL, threadpool_worker, threadpool);
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

  free(threadpool->workers);
  free(threadpool);
}

void threadpool_wait(threadpool_t *threadpool) {

  pthread_mutex_lock(&(threadpool->mutex));
  while (1) {
    if (threadpool->work_begin != NULL ||
        (!threadpool->stop && threadpool->working_cnt != 0)) {
      pthread_cond_wait(&(threadpool->working_cond), &(threadpool->mutex));
    } else {
      break;
    }
  }
  pthread_mutex_unlock(&(threadpool->mutex));
}

bool threadpool_add_work(threadpool_t *threadpool, thread_func_t func,
                         void *args) {

  threadpool_work_t *new_work = threadpool_work_create(func, args);
  if (!new_work)
    return false;

  pthread_mutex_lock(&threadpool->mutex);
  if (!threadpool->work_begin) {
    threadpool->work_begin = new_work;
  } else {
    threadpool->work_end->next = new_work;
  }
  threadpool->work_end = new_work;

  pthread_cond_broadcast(&(threadpool->work_cond));
  pthread_mutex_unlock(&(threadpool->mutex));

  return true;
}
