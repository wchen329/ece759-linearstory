#ifndef __LU_KERNEL_CUH__
#define __LU_KERNEL_CUH__

#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

#include <stdio.h>

#include "device_launch_parameters.h"

namespace linearstory
{
	template<class DataType>
	__global__ void lu_decompose(
		DataType* A,
		DataType* B,
		DataType* X,
		DataType* S,
		DataType* S_tmp,
		DataType* op,
		DataType* l_buffer,
		DataType* L,
		DataType* U,
		size_t dim_full,
		size_t ind
	)
	{
		// Decompose using iteration (this only decomposes for one iteration).
		int ordinal = threadIdx.x + blockDim.x * blockIdx.x;

		// Careful here... this is kind of bad because 1 thread really does very little and then stops
		// can make it sequential instead...
		if(ordinal ==  0)
		{
			L[ind * dim_full + ind] = 1;
			U[ind * dim_full + ind] = S[0];
		}
		else
		{
		}

		__syncthreads();
		

		// The zero ordinal is no more
		ordinal++;

		{
			// For all u in U | u has y = ind, u = a
			size_t col_flat = ordinal;
			unsigned incr = blockDim.x * gridDim.x;
			unsigned dim_S = dim_full - ind;
			for (size_t col = ind + ordinal; col < dim_full; col += incr)
			{
				U[ind * dim_full + col] = S[col_flat];
				col_flat += incr;
			}

			__syncthreads();

			// For all l in L | l has x = ind, l = a/u
			size_t row_flat = ordinal;
			for (size_t row = ind + ordinal; row < dim_full; row += incr)
			{
				l_buffer[row_flat - 1] = S[row_flat * dim_S] / S[0];
				L[(row) * dim_full + ind] = l_buffer[row_flat - 1];
				row_flat += incr;
			}

			__syncthreads();

			// Calculate S
			// copy the submatrix of A[1:,1:] into S. Reorigin it.
			for (size_t sy = ordinal; sy < dim_S; sy += incr)
			{
				for (size_t sx = 1; sx < dim_S; ++sx)
				{
					size_t sy_out = (sy - 1);
					size_t sx_out = (sx - 1);
					size_t raw_index = sy_out * dim_S + sx_out - (sy - 1);
					S_tmp[raw_index] = S[sy * dim_S + sx];
				}
			}

			__syncthreads();

			// Calculate the final value of S.
			DataType* lptr = l_buffer;
			DataType* uptr = U + (ind * dim_full) + ind + 1;
			DataType* hopptr = op;
			unsigned rank = dim_S - 1;

			// Calculate the outer product
			for (size_t i = ordinal - 1; i < rank; i += incr)
			{
				for (size_t j = 0; j < rank; ++j)
				{
					hopptr[i * rank + j] = lptr[i] * uptr[j];
				}
			}

			__syncthreads();

			// Perform a matrix-matrix subtract
			for (size_t i = ordinal - 1; i < rank; i += incr)
			{
				for (size_t j = 0; j < rank; ++j)
				{
					size_t loc = i * rank + j;
					S[loc] = S_tmp[loc] - hopptr[loc];
				}
			}

			__syncthreads();
		}
	}

	// Unusable, incorrect kernel.
	template<class DataType>
	__global__ void lu_forward_sub(DataType* L, DataType* B, DataType * k, size_t dim_pvt)
	{
		int ordinal = threadIdx.x + blockDim.x * blockIdx.x;

		int iter = blockDim.x * gridDim.x;
		for (size_t y = ordinal; y < dim_pvt; y += iter)
		{
			DataType val = B[y];

			// Solve L and put the result into k
			for (size_t x = 0; x < y; ++x)
			{
				// RAW Hazard... not usable
				val -= L[y * dim_pvt + x] * k[x];
			}

			k[y] = val / L[y * dim_pvt + y];
		}

		__syncthreads();
	}


	// Unusable, incorrect kernel.
	template<class DataType>
	__global__ void lu_back_sub(DataType* U, DataType* k, DataType* x, size_t dim_pvt)
	{
		// x is already zero filled
		int ordinal = threadIdx.x + blockDim.x * blockIdx.x;

		int iter = blockDim.x * gridDim.x;
		for (size_t y = ordinal; y < dim_pvt; y += iter)
		{
			DataType val = k[y];

			// Solve U and put the result into x
			for (size_t x_c = dim_pvt - 1; x_c > y; --x_c)
			{
				// RAW Hazard... not usable
				val -= U[y * dim_pvt + x_c] * x[x_c];
			}

			x[y] = val / U[y * dim_pvt + y];
		}
	}
}

#endif