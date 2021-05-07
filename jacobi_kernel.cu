// #include <__clang_cuda_builtin_vars.h>
#include <cstdio>
#include <iostream>
#include <cuda.h>
#include "jacobi_kernel.cuh"
#include <math.h>
#include <vector>
 
void print_arr(float* arr, int n)
{
    for (int i = 0; i < n; i++) printf("%f ", arr[i]);
    printf("\n");
}

__global__ void jacobi_kernel(float *A, float *B, float *x, float *x_new, int n)
{
	int tx = blockIdx.x*blockDim.x + threadIdx.x;
	if (tx >= n) return;


	extern __shared__ float shared_memory[];
	float *shared_x = shared_memory;
	float *shared_x_new = shared_x + blockDim.x;
	float *shared_B = shared_x_new + blockDim.x;

	shared_x[threadIdx.x] = x[tx];
	shared_B[threadIdx.x] = B[tx];
	__syncthreads();

	float LU_sum = 0;
	
	for (int j = 0; j < n; j++)
	{
		LU_sum += A[tx*n + j]*x[j];
		// LU_sum += A[tx*n + j]*x[j];
	}
	LU_sum -= A[tx*n+tx]*x[threadIdx.x];
	shared_x_new[threadIdx.x] = (shared_B[threadIdx.x] - LU_sum)/A[tx*n + tx];
	// LU_sum -= A[tx*n+tx]*x[tx];
	// x_new[tx] = (B[tx] - LU_sum)/A[tx*n + tx];

	__syncthreads();
	x_new[tx] = shared_x_new[threadIdx.x];


}

void jacobi(float *A, float *B, float *x, int n, int threads_per_block)
{
	const int ITERATION_LIMIT = 200;

	int numBlocks = (int)ceil( (double)n/(double)threads_per_block);
	int shared_space = sizeof(float)*3*threads_per_block; // for x, x_new, and B
	
	float *dA, *dB, *dx, *dx_new;

	cudaMalloc((void**)&dA, sizeof(float) * n * n);
	cudaMalloc((void**)&dB, sizeof(float) * n);
	cudaMalloc((void**)&dx, sizeof(float) * n);
	cudaMalloc((void**)&dx_new, sizeof(float) * n);

	cudaMemcpy(dA, A, sizeof(float) * n * n, cudaMemcpyHostToDevice);
	cudaMemcpy(dB, B, sizeof(float) * n, cudaMemcpyHostToDevice);
	cudaMemcpy(dx, x, sizeof(float) * n, cudaMemcpyHostToDevice);

	cudaEvent_t start;
	cudaEvent_t stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	
	cudaEventRecord(start);

	for (int k = 0; k < ITERATION_LIMIT/2; k++)
    {
		jacobi_kernel<<<numBlocks, threads_per_block, shared_space>>>(dA, dB, dx, dx_new, n);
		jacobi_kernel<<<numBlocks, threads_per_block, shared_space>>>(dA, dB, dx_new, dx, n);

		// cudaMemcpy(dx, dx_new, sizeof(float) * n, cudaMemcpyDeviceToDevice);
	}
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);

	float ms;
	cudaEventElapsedTime(&ms, start, stop);

	cudaMemcpy(x, dx, sizeof(float) * n, cudaMemcpyDeviceToHost);

	printf("ms: %f \n", ms);
    // printf("Result: ");
	// print_arr(x, n);

	cudaFree(dA);
	cudaFree(dB);
	cudaFree(dx);
	cudaFree(dx_new);

	return;
}

