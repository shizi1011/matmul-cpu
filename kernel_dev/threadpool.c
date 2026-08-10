#include "threadpool.h"
#include <pthread.h>
#include <stdlib.h>

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

  size_t working_cnt; // how many threads are actively processing work
  size_t thread_cnt;  // how many threads are alive
  bool stop;          // stop threads
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

    if (threadpool->stop)
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
  threadpool->thread_cnt--;
  pthread_cond_signal(&(threadpool->working_cond));
  pthread_mutex_unlock(&(threadpool->mutex));
  return NULL;
}

threadpool_t *threadpool_create(size_t num) {
  threadpool_t *threadpool;
  pthread_t thread;
  size_t i;

  if (num == 0)
    num = 2;

  threadpool = calloc(1, sizeof(*threadpool));
  threadpool->thread_cnt = num;

  pthread_mutex_init(&(threadpool->mutex), NULL);
  pthread_cond_init(&(threadpool->work_cond), NULL);
  pthread_cond_init(&(threadpool->working_cond), NULL);

  threadpool->work_begin = NULL;
  threadpool->work_end = NULL;

  for (i = 0; i < num; i++) {
    pthread_create(&thread, NULL, threadpool_worker, threadpool);
    pthread_detach(thread);
  }

  return threadpool;
}

void threadpool_destroy(threadpool_t *threadpool) {
  threadpool_work_t *work;
  threadpool_work_t *work2;

  if (threadpool == NULL)
    return;

  pthread_mutex_lock(&(threadpool->mutex));
  work = threadpool->work_begin;
  while (work != NULL) {
    work2 = work->next;
    threadpool_work_destroy(work);
    work = work2;
  }
  threadpool->work_begin = NULL;
  threadpool->stop = true;
  pthread_cond_broadcast(&(threadpool->work_cond));
  pthread_mutex_unlock(&(threadpool->mutex));

  threadpool_wait(threadpool);

  pthread_mutex_destroy(&(threadpool->mutex));
  pthread_cond_destroy(&(threadpool->work_cond));
  pthread_cond_destroy(&(threadpool->working_cond));

  free(threadpool);
}

void threadpool_wait(threadpool_t *threadpool) {
  if (threadpool == NULL)
    return;

  pthread_mutex_lock(&(threadpool->mutex));
  while (1) {
    if (threadpool->work_begin != NULL ||
        (!threadpool->stop && threadpool->working_cnt != 0) ||
        (threadpool->stop && threadpool->thread_cnt != 0)) {
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
