#pragma once

#include <stdbool.h>

void kernel_16x6_accum(float *blockA_packed, float *blockB_packed, float *C,
                       int mr, int nr, int kc, int M, bool load);
