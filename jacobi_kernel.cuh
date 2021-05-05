#ifndef JACOBI_CUH
#define JACOBI_CUH

__global__ void jacobi_kernel(float *A, float *B, float *x, float *x_new, int n);

// The kernel call should be followed by a call to cudaDeviceSynchronize for timing purposes.
void jacobi(float *A, float *B, float *x, int n, int threads_per_block);

#endif
