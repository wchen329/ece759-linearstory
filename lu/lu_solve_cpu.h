#ifndef __LU_SOLVER_CPU_H___
#define __LU_SOLVER_CPU_H__
#include <algorithm>
#include <cassert>
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
			 * Makes L and U out of A (Using Leading Row-Colum LU)
			 */
			virtual void Decompose() 
			{
				// S: this is a temporary which gets reconstructed every iteration of dim_offset 
				DataType* S = host_S.get();
				// S already initalized to A

				// Decompose using iteration 
				for(size_t dim_offset = 0; dim_offset < dim_pvt; ++dim_offset)
				{
					// Set pivots of L and U
					DataType* L = host_L.get();
					DataType* U = host_U.get();

					size_t dim_S = dim_pvt - dim_offset;

					// Constant Time
					L[dim_offset + dim_pvt * dim_offset] = 1;
					U[dim_offset + dim_pvt * dim_offset] = S[dim_offset * dim_S + dim_offset];

					// Set this row (dim_offset) and this column to
					// be the correct values
					
					// For all u in U | u has y = dim_offset, u = a
					size_t col_flat = 1;
					for(size_t col = dim_offset + 1; col < dim_pvt; ++col)
					{
						//U[dim_offset * dim_pvt + col] = S[dim_offset * dim_S + col];
						U[dim_offset * dim_pvt + col] = S[col_flat];
						++col_flat;
					}

					// For all l in L | l has x = dim_offset, l = a/u
					size_t row_flat = 1;
					for(size_t row = dim_offset + 1; row < dim_pvt; ++row)
					{
						//l_buffer.get()[row] = S[row * dim_S + dim_offset] / U[dim_offset * dim_pvt + row];
						l_buffer.get()[row] = S[(row) * dim_S] / U[dim_offset * dim_pvt + row];
						L[row * dim_pvt + dim_offset] = l_buffer.get()[row_flat];
						++row_flat;
					}

					#ifdef VERBOSE_DEBUG
						MatEcho<DataType>(S, dim_S, dim_S);	
					#endif

					// Calculate S
					// copy the submatrix of A[1:,1:] into S. Reorigin it.
					size_t counter = 0;
					for(size_t sy = 1; sy < dim_S; ++sy)
					{
						for(size_t sx = 1; sx < dim_S; ++sx)
						{
							host_S_tmp.get()[counter] = S[sy * dim_S + sx];
							++counter;
						}
					}

					#ifdef VERBOSE_DEBUG
						//MatEcho<DataType>(host_S_tmp.get(), dim_S - 1, dim_S - 1);	
					#endif

					// Calculate the final value of S.
					DataType* lptr = l_buffer.get() + dim_offset;
					DataType* uptr = U + (dim_offset * dim_pvt);
					DataType* hopptr = host_op.get();
					mat_outer_product(lptr, uptr, hopptr, dim_S - 1);

					#ifdef VERBOSE_DEBUG
						MatEcho<DataType>(lptr, 1, dim_S - 1);	
						MatEcho<DataType>(uptr, 1, dim_S - 1);	
						assert(dim_offset != 1);
					#endif

					// Do matrix subtraction of S_tmp and outer_product
					mat_sub(host_S_tmp.get(), hopptr, S, dim_S - 1);
				}
				
			}

			void mat_outer_product(DataType* a, DataType* b, DataType* output, size_t rank)
			{
				// Calculate the outer product
				for(size_t i = 0; i < rank; ++i)
				{
					for(size_t j = 0; j < rank; ++j)
					{
						output[i * rank + j] = a[i] * b[j];
					}
				}
			}

			void mat_sub(DataType * A, DataType* B, DataType* output, size_t rank)
			{
				for(size_t i = 0; i < rank; ++i)
				{
					for(size_t j = 0; j < rank; ++j)
					{
						size_t loc = i * rank + j;
						output[loc] = A[loc] - B[loc];
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
					MatEcho<DataType>(host_S.get(), dim_pvt, dim_pvt);
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
				host_S_tmp(new DataType[dim * dim]),
				host_op(new DataType[dim * dim]),
				host_k(new DataType[dim]),
				l_buffer(new DataType[dim]),
				LinearSystem<DataType>(dim)
			{
				// Zero Host arrays L and U
				std::fill_n<DataType*>(host_L.get(), dim * dim, 0);
				std::fill_n<DataType*>(host_U.get(), dim * dim, 0);

				// Start S as a copy of A
				std::copy<DataType*>(LinearSystem<DataType>::get1D_A_Host()
,
					 LinearSystem<DataType>::get1D_A_Host() + (dim_pvt * dim_pvt), host_S.get());

			}

		protected:
			MArray host_L;
			MArray host_U;
			MArray host_S;
			MArray host_S_tmp;
			MArray host_k;
			MArray host_op;
			MArray l_buffer;
		private:
			size_t dim_pvt;
	};
}

#endif

