#!/bin/bash

MINSIZE=20000;
STEPSIZE=2000;
NPTS=5;
WNITER=0;
NITER_START=1;
NITER_END=1;
SAVEDIR="benchmark_data"

# rm -r $PWD/build
# cmake -B $PWD/build -S $PWD -DNTHREADS=${1} -DOMP_SCHEDULE=${2}
# cmake --build $PWD/build -t benchmark
$PWD/build/benchmark ${MINSIZE} ${STEPSIZE} ${NPTS} ${WNITER} ${NITER_START} ${NITER_END} ${SAVEDIR}

