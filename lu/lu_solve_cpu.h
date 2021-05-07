#ifndef __LU_SOLVER_CPU_H___
#define __LU_SOLVER_CPU_H__
#include <algorithm>
#include <cstdint>
#include "matecho.h"
#include "linsys.cuh"

namespace linearstory
{

	// Exception class
	class NoSolutionFound
	{
	};

	template<class DataType>
	class LUSystem_CPU: public LinearSystem<DataType>
	{
		public:
			/* Decompose
			 * Makes L and U out of A
			 */
			virtual void Decompose() 
			{
				// Decompose using iteration 
				for(size_t dim_offset = 0; dim_offset < dim_pvt; ++dim_offset)
				{
					// Set pivots of L and U
					size_t radial_offset = dim_pvt * dim_offset  + dim_offset;

					DataType* L = host_L.get();
					DataType* U = host_U.get();

					L[dim_offset + dim_pvt * dim_offset] = 1;
					U[dim_offset + dim_pvt * dim_offset] = LinearSystem<DataType>::atA(dim_offset, dim_offset);

					// Set this row (dim_offset) and this column to
					// be the correct values
					
					// For all u in U | u has y = dim_offset, u = a
					for(size_t col = dim_offset + 1; col < dim_pvt; ++col)
					{
						U[dim_offset * dim_pvt + col] = LinearSystem<DataType>::atA(dim_offset, col);
					}

					// For all l in L | l has x = dim_offset, l = a/u
					for(size_t row = dim_offset + 1; row < dim_pvt; ++row)
					{
						// Bad locality... might want to optimize by creating a transpose version of A as well.
						L[row * dim_pvt + dim_offset] = LinearSystem<DataType>::atA(row, dim_offset) / U[dim_offset * dim_pvt + row];
					}

				}
				
			}

			/* forward substitution
			 * Solve a lower triangular matrix (L), and put the results in host_k
			 */
			virtual void forward_sub()
			{
				DataType* k = host_k.get();
				std::fill_n(k, dim_pvt, 0);

				for(size_t y = 0; y < dim_pvt; ++y)
				{
					DataType val = LinearSystem<DataType>::atB(y);

					// Solve L and put the result into k
					for(size_t x = 0; x < y; ++x)
					{
						val -= host_L.get()[y * dim_pvt + x] * k[x];
					}

					k[y] = val / host_L.get()[y * dim_pvt + y];
				}
			}

			/* backward substituion
			 * Solve an upper triangular matrix (K), and put the results in host_x (base class buffer)
			 */
			virtual void backward_sub()
			{
				DataType* x_arr = LinearSystem<DataType>::get1D_X_Host();
				// x is already zero filled

				for(size_t y = dim_pvt - 1; y > 0; --y)
				{
					DataType val = host_k.get()[y];

					// Solve U and put the result into x
					for(size_t x = dim_pvt - 1; x > y; --x)
					{
						val -= host_U.get()[y * dim_pvt + x] * x_arr[x];
					}

					x_arr[y] = val / host_U.get()[y * dim_pvt + y];
				}

				// Unroll the last iteration, due to underflow
				DataType val = host_k.get()[0];

				// Solve U and put the result into x
				for(size_t x = dim_pvt - 1; x > 0; --x)
				{
					val -= host_U.get()[x] * x_arr[x];
				}

				x_arr[0] = val / host_U.get()[0];
			}

			/* Solve
			 * Decompose A into L and U. Then,
			 * use forward and backward substituion to solve for intermediate
			 * "k" and then "x" respectively
			 */
			virtual void solve()
			{
				#ifdef VERBOSE_DEBUG
					MatEcho<DataType>(LinearSystem<DataType>::get1D_A_Host(), dim_pvt, dim_pvt);
					MatEcho<DataType>(LinearSystem<DataType>::get1D_B_Host(), 1, dim_pvt);
				#endif

				// Perform decomposition
				Decompose();

				#ifdef VERBOSE_DEBUG
					MatEcho<DataType>(host_L.get(), dim_pvt, dim_pvt);
					MatEcho<DataType>(host_U.get(), dim_pvt, dim_pvt);
				#endif

				// Get k
				forward_sub();

				#ifdef VERBOSE_DEBUG
					MatEcho<DataType>(host_k.get(), 1, dim_pvt);
				#endif
				
				// Get x
				backward_sub();
				#ifdef VERBOSE_DEBUG
					MatEcho<DataType>(LinearSystem<DataType>::get1D_X_Host(), 1, dim_pvt);
				#endif
			}

			typedef std::unique_ptr<DataType, std::default_delete<DataType[]>> MArray;
			LUSystem_CPU<DataType>(size_t dim) :
				dim_pvt(dim),
				host_L(new DataType[dim * dim]),
				host_U(new DataType[dim * dim]),
				host_S(new DataType[dim * dim]),
				host_k(new DataType[dim]),
				LinearSystem<DataType>(dim)
			{
				// Zero Host arrays L and U
				std::fill_n<DataType*>(host_L.get(), dim * dim, 0);
				std::fill_n<DataType*>(host_U.get(), dim * dim, 0);

			}

		protected:
			MArray host_L;
			MArray host_U;
			MArray host_S;
			MArray host_k;
		private:
			size_t dim_pvt;
	};
}

#endif

