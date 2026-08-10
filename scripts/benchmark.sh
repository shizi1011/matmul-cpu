#!/bin/bash

MINSIZE=500;
STEPSIZE=500;
NPTS=10;
WNITER=0;
NITER_START=15;
NITER_END=5;
SAVEDIR="benchmark_data"

# rm -r $PWD/build
# cmake -B $PWD/build -S $PWD -DNTHREADS=${1} -DOMP_SCHEDULE=${2}
# cmake --build $PWD/build -t benchmark
$PWD/build/benchmark ${MINSIZE} ${STEPSIZE} ${NPTS} ${WNITER} ${NITER_START} ${NITER_END} ${SAVEDIR}

