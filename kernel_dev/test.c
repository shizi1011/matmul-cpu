#include <pthread.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

#include "threadpool.h"

static const size_t num_threads = 4;
static const size_t num_items = 100;

void worker(void *arg) {
  int *val = arg;
  int old = *val;

  *val += 1000;
  printf("tid=%p, old=%d, val=%d\n", pthread_self(), old, *val);

  if (*val % 2)
    usleep(100000);
}

int main(int argc, char **argv) {
  threadpool_t *threadpool;
  int *vals;
  size_t i;

  threadpool = threadpool_create(num_threads);
  vals = calloc(num_items, sizeof(*vals));

  for (i = 0; i < num_items; i++) {
    vals[i] = i;
    threadpool_add_work(threadpool, worker, vals + i);
  }

  threadpool_wait(threadpool);

  for (i = 0; i < num_items; i++) {
    printf("%d\n", vals[i]);
  }

  free(vals);
  threadpool_destroy(threadpool);
  return 0;
}
