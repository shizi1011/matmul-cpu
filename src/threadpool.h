#pragma once

// #include <pthread.h>
#include <stdbool.h>
#include <stddef.h>

struct threadpool;
typedef struct threadpool threadpool_t;

// struct threadpool_work;
// typedef struct threadpool_work threadpool_work_t;

typedef void (*thread_func_t)(void *args);

threadpool_t *threadpool_create(size_t num_threads, size_t queue_capacity);
void threadpool_destroy(threadpool_t *threadpool);

bool threadpool_add_work(threadpool_t *threadpool, thread_func_t func,
                         void *args);
void threadpool_wait(threadpool_t *threadpool);

// struct threadpool_work {
//   thread_func_t func;
//   void *args;
//   struct threadpool_work *next;
// };
