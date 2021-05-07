#ifndef JACOBI_OMP_H
#define JACOBI_OMP_H

#include <cstddef>
#include <omp.h>

void print_arr(float* arr, int n);
void jacobi_omp(float *A, float *B, float *x, int n);
void matmul(float* A, float* x, float* B_test, int n);
void verify(float* B, float* test, int n);

#endif