#ifndef __LU_SOLVE_CUDA_CUH__
#define __LU_SOLVE_CUDA_CUH__

#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>
#include "cudamemshuttle.cuh"
#include "linsys.cuh"
#include "lu_kernel.cuh"
#include "matecho.h"

namespace linearstory
{
	template<class DataType>
	class LU_CUDA_System : public LinearSystem<DataType>
	{
		public:
			
			/* forward substitution
			* Solve a lower triangular matrix (L), and put the results in host_k
			*/
			virtual void forward_sub(DataType* host_k, DataType* host_L)
			{
				DataType* k = host_k;

				for (size_t y = 0; y < dim_pvt; ++y)
				{
					DataType val = LinearSystem<DataType>::atB(y);

					// Solve L and put the result into k
					for (size_t x = 0; x < y; ++x)
					{
						val -= host_L[y * dim_pvt + x] * k[x];
					}

					k[y] = val / host_L[y * dim_pvt + y];
				}
			}

			/* backward substituion
			* Solve an upper triangular matrix (K), and put the results in host_x (base class buffer)
			*/
			virtual void backward_sub(DataType* host_k, DataType* host_U)
			{
				DataType* x_arr = LinearSystem<DataType>::get1D_X_Host();
				// x is already zero filled

				for (size_t y = dim_pvt - 1; y > 0; --y)
				{
					DataType val = host_k[y];

					// Solve U and put the result into x
					for (size_t x = dim_pvt - 1; x > y; --x)
					{
						val -= host_U[y * dim_pvt + x] * x_arr[x];
					}

					x_arr[y] = val / host_U[y * dim_pvt + y];
				}

				// Unroll the last iteration, due to underflow
				DataType val = host_k[0];

				// Solve U and put the result into x
				for (size_t x = dim_pvt - 1; x > 0; --x)
				{
					val -= host_U[x] * x_arr[x];
				}

				x_arr[0] = val / host_U[0];
			}

			// Solve this system
			void solve()
			{
				// Managed Host Pointer Type
				typedef std::unique_ptr<DataType, std::default_delete<float[]>> dtp;

#ifdef VERBOSE_DEBUG
				dtp a(new DataType[dim_pvt * dim_pvt]);
				device_A.pullHostArr(a.get());
				MatEcho<float>(a.get(), dim_pvt, dim_pvt);
#endif

				// Perform decompose
				for (size_t itr = 0; itr < dim_pvt; ++itr)
				{
#ifdef VERBOSE_DEBUG
					// Debug: Print S
					dtp s_b(new DataType[dim_pvt * dim_pvt]);
					device_S.pullHostArr(s_b.get());
					MatEcho<DataType>(s_b.get(), dim_pvt - itr , dim_pvt - itr);
#endif

					lu_decompose<DataType> <<<10, threads_per_block >>>(
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

#ifdef VERBOSE_DEBUG
					// Debug: Print S
					dtp s(new DataType[dim_pvt * dim_pvt]);
					device_S.pullHostArr(s.get());
					MatEcho<DataType>(s.get(), dim_pvt - itr - 1, dim_pvt - itr - 1);

					dtp s_tmp(new DataType[dim_pvt * dim_pvt]);
					device_S_tmp.pullHostArr(s_tmp.get());
					MatEcho<DataType>(s_tmp.get(), dim_pvt - itr, dim_pvt - itr);

					dtp lbuf(new DataType[dim_pvt]);
					device_lbuffer.pullHostArr(lbuf.get());
					MatEcho<DataType>(lbuf.get(), 1, dim_pvt - itr - 1);

					dtp l(new DataType[dim_pvt * dim_pvt]);
					device_L.pullHostArr(l.get());
					MatEcho<DataType>(l.get(), dim_pvt, dim_pvt);

					dtp u(new DataType[dim_pvt * dim_pvt]);
					device_U.pullHostArr(u.get());
					MatEcho<DataType>(u.get(), dim_pvt, dim_pvt);

					dtp op(new DataType[dim_pvt * dim_pvt]);
					device_op.pullHostArr(op.get());
					MatEcho<DataType>(op.get(), dim_pvt - itr, dim_pvt - itr);

					fprintf(stdout, "-----------------NEW ITERATION------------------\n");
#endif

				}

				cudaDeviceSynchronize();

				dtp l(new DataType[dim_pvt * dim_pvt]);
				device_L.pullHostArr(l.get());
				dtp u(new DataType[dim_pvt * dim_pvt]);
				device_U.pullHostArr(u.get());
				dtp k(new DataType[dim_pvt]);
				device_k.pullHostArr(k.get());
				forward_sub(k.get(), l.get());
				backward_sub(k.get(), u.get());

#ifdef VERBOSE_DEBUG
				dtp k_b(new DataType[dim_pvt]);
				device_k.pullHostArr(k_b.get());
				MatEcho<DataType>(k_b.get(), 1, dim_pvt);
				MatEcho<DataType>(LinearSystem<DataType>::get1D_X_Host(), 1, dim_pvt);
#endif
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