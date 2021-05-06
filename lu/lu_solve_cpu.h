#ifndef __LU_SOLVER_CPU_H___
#define __LU_SOLVER_CPU_H__
#include <algorithm>
#include <cstdint>
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
				for(size_t dim_remain = dim_pvt; dim_remain > 0; --dim_remain)
				{
					// Set pivots of L and U
					size_t dim_offset = dim_pvt - dim_remain;
					size_t radial_offset = dim_pvt * dim_offset  + dim_offset;

					DataType* L = host_L.get() + radial_offset;
					DataType* U = host_U.get() + radial_offset;

					L[0] = 1;
					U[0] = LinearSystem<DataType>::atA(0,0);

					// Set this row (dim_offset) and this column to
					// be the correct values
					
					// For all u in U | u has y = dim_offset, u = a
					for(size_t col = 0; col < dim_pvt; ++col)
					{
						U[dim_offset * dim_pvt + col] = LinearSystem<DataType>::atA(dim_offset, col);
					}

					// For all l in L | l has x = dim_offset, l = a
					for(size_t row = 0; row < dim_pvt; ++row)
					{
						// Bad locality... might want to optimize by creating a transpose version of A as well.
						L[row * dim_pvt + dim_offset] = LinearSystem<DataType>::atA(row, dim_offset);
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

					// Solve L and put the result into k (going backwards)
					for(size_t x = y; x < dim_pvt; ++x)
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

				for(size_t y = dim_pvt - 1; y >= 0; --y)
				{
					DataType val = host_k.get()[y];

					// Solve U and put the result into x
					for(size_t x = y; x < dim_pvt; ++x)
					{
						val -= host_U.get()[y * dim_pvt + x] * x_arr[x];
					}

					x_arr[y] = val / host_L.get()[y * dim_pvt + y];
				}
			}

			/* Solve
			 * Decompose A into L and U. Then,
			 * use forward and backward substituion to solve for intermediate
			 * "k" and then "x" respectively
			 */
			virtual void solve()
			{
				// Perform decomposition
				Decompose();

				// Get k
				forward_sub();
				
				// Get x
				backward_sub();
			}

			typedef std::unique_ptr<DataType, std::default_delete<DataType[]>> MArray;
			LUSystem_CPU<DataType>(size_t dim) :
				dim_pvt(dim),
				host_L(new DataType[dim * dim]),
				host_U(new DataType[dim * dim]),
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
			MArray host_k;
		private:
			size_t dim_pvt;
	};
}

#endif
