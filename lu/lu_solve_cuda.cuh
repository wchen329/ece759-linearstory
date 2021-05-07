#ifndef __LU_SOLVE_CUDA_CUH__
#define __LU_SOLVE_CUDA_CUH__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cudamemshuttle.cuh"
#include "linsys.cuh"
#include "lu_kernel.cuh"

namespace linearstory
{
	template<class DataType>
	class LU_CUDA_System : public LinearSystem<DataType>
	{
		public:
			
			// Solve this system
			void solve()
			{
				// Perform decompose
				for (size_t itr = 0; itr < dim_pvt; ++itr)
				{
					lu_decompose<DataType> <<<1, threads_per_block >>>(
						device_A.raw(),
						device_B.raw(),
						device_X.raw(),
						device_S.raw(),
						device_S_tmp.raw(),
						device_op.raw(),
						device_lbuffer.raw(),
						device_L.raw(),
						device_U.raw(),
						dim_pvt,
						itr
					);
				}
				cudaDeviceSynchronize();
				

				// Perform forward substition
				lu_forward_sub<DataType> << <1, threads_per_block >> > (
					device_L.raw(),
					device_B.raw(),
					device_k.raw(),
					dim_pvt
				);
				cudaDeviceSynchronize();
				lu_back_sub<DataType> << <1, threads_per_block >> > (
					device_U.raw(),
					device_k.raw(),
					device_X.raw(),
					dim_pvt
					);
				// Perform backward substituion
				
				cudaDeviceSynchronize();

				// Get our values of X from our GPU
				device_X.pullHostArr(LinearSystem<DataType>::get1D_X_Host());
			}

			LU_CUDA_System(size_t dim, size_t threads_per_block_in) :
				dim_pvt(dim),
				LinearSystem<DataType>(dim),
				threads_per_block(threads_per_block_in),
				device_A(dim * dim),
				device_B(dim),
				device_X(dim),
				device_L(dim * dim),
				device_U(dim * dim),
				device_S(dim * dim),
				device_S_tmp(dim * dim),
				device_k(dim),
				device_op(dim * dim),
				device_lbuffer(dim)
			{
				device_A.pushHostArr(LinearSystem<DataType>::get1D_A_Host());
				device_B.pushHostArr(LinearSystem<DataType>::get1D_B_Host());
				device_X.zero();
				device_k.zero();
				device_S.pushHostArr(LinearSystem<DataType>::get1D_A_Host());
				device_L.zero();
				device_U.zero();
			}

		private:
			size_t dim_pvt;
			size_t threads_per_block;
			CudaMemoryShuttle<DataType> device_A;
			CudaMemoryShuttle<DataType> device_B;
			CudaMemoryShuttle<DataType> device_X;
			CudaMemoryShuttle<DataType> device_L;
			CudaMemoryShuttle<DataType> device_U;
			CudaMemoryShuttle<DataType> device_S;
			CudaMemoryShuttle<DataType> device_S_tmp;
			CudaMemoryShuttle<DataType> device_k;
			CudaMemoryShuttle<DataType> device_op;
			CudaMemoryShuttle<DataType> device_lbuffer;
	};
}

#endif