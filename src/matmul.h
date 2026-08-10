#pragma once

void matmul_openmp(float *A, float *B, float *C, int M, int N, int K);
void matmul_threadpool(float *A, float *B, float *C, int M, int N, int K);
void matmul_naive(float *A, float *B, float *C, int m, int n, int k);
