#ifndef __LINSYS_CUH__
#define __LINSYS_CUH__
#include <algorithm>
#include <cctype>
#include <cstdint>
#include <iostream>
#include <memory>
#include <random>

#include "cuda_mock.h"
#include "randomgen.h"

namespace linearstory
{

	/* Linear System: Base Class
	 * A linear system is simply a matrix of several linear functions' coefficients
	 * Several operations can be done over it
	 * Indexed in row major order
	 *
	 * Parameters: n - number of rows/cols, length of x, length of b
	 *
	 * NOTE: This type can not be copied, as it contains
	 * unique reference to data types
	 */

	typedef uint32_t Coeff;

	template<class DataType> class LinearSystem
	{
		public:

			/* Row
			 * A row in the linear system
			 */
			typedef std::unique_ptr<DataType, std::default_delete<DataType[]>> HostArray;
			typedef DataType* RowPtr;
			typedef DataType* RawArray;

			/* operator[]
			 * Index the underlying array, 2D semantics
			 */
			RowPtr operator[](uint32_t ind)
			{
				return &(host_a.get()[ind * dim]);
			}

			const RowPtr operator[](uint32_t ind) const
			{
				return &(host_a.get()[ind * dim]);
			}

			DataType& atA(uint32_t y, uint32_t x)
			{
				return host_a.get()[(y * dim) + x];
			}

			DataType& atX(uint32_t y)
			{
				return host_x.get()[y];
			}

			DataType& atB(uint32_t y)
			{
				return host_b.get()[y];
			}

			/* verify
			 * Return "true" if x is valid for the given system, else return "false"
			 */
			bool verify()
			{
				HostArray scratch_output(new DataType[dim]);
				std::fill_n<RawArray>(scratch_output.get(), dim, 0);

				// Try matmul'ing to recreate the output (seq).
				for(uint64_t din = 0; din < dim; ++din)
				{
					DataType dot_prod = 0;

					for(uint64_t dij = 0; dij < dim; ++ dij)
					{
						dot_prod += (*this)[din][dij] * host_x.get()[dij];
					}

					scratch_output.get()[din] = dot_prod;
				}


				bool all_correct = true;
				for(uint64_t di2 = 0; di2 < dim; ++di2)
				{
					/* Hack!
					 * Use runtime type checking to change equality function
					 */

					if(typeid(scratch_output.get()[0]) == typeid(float))
					{
						float err = (scratch_output.get()[di2] - host_b.get()[di2]);

						// OK if less than 0.00 1 (0.1%) error
						if(fabs(err) > 0.001)
						{
							std::cout <<  "[verify] b["  << di2 << "] is INCORRECT. Got Value: " << scratch_output.get()[di2]
								 << "; Expected Value: " << host_b.get()[di2] << ";" << std::endl;
							all_correct = false;
						}
						else
						{
							std::cout <<  "[verify] b["  << di2 << "] is correct. Value: " << host_b.get()[di2] << std::endl;
						}
					}
					else
					{
						if(scratch_output.get()[di2] != host_b.get()[di2])
						{
							std::cout <<  "[verify] b["  << di2 << "] is INCORRECT. Got Value: " << scratch_output.get()[di2]
								 << "; Expected Value: " << host_b.get()[di2] << ";" << std::endl;
							all_correct = false;
						}
						else
						{
							std::cout <<  "[verify] b["  << di2 << "] is correct. Value: " << host_b.get()[di2] << std::endl;
						}
					}
				}

				return all_correct;
			}

			/* solve
		 	 * Solve this linear system (to be reimplemented according to the corresponding algorithm)
			 * When this is called, the device array which represents this system, will be copied
			 * This will return another allocated array in host memory.
			 */
			virtual void solve() = 0;

			/* LinearSystem
			 * Default Class, create an array in host memory, and create the corresponding Device Array
			 *
			 * Use instantiation by random device
			 */
			LinearSystem<DataType>(size_t n) :
				dim(n),
				host_a(new DataType[n*n]),
				host_x(new DataType[n]),
				host_b(new DataType[n]),
				lsr()
			{
				for(size_t ind = 0; ind < (n*n); ++ind)
				{
					host_a.get()[ind] = lsr(); 
				}
				for(size_t ind = 0; ind < n; ++ind)
				{
					host_b.get()[ind] = lsr(); 
				}
				std::fill_n<RawArray>(host_x.get(), dim, 0);
			}
	
		protected:
			/* RawArray get1D_A (host)
			 * Returns the handle to the 1D representation of A. 
			 */
			 RawArray get1D_A_Host() { return this->host_a.get(); }

			/* RawArray get1D_X (host)
			 * Returns the handle to the 1D representation of B.
			 */
			 RawArray get1D_X_Host() { return this->host_x.get(); }

			/* RawArray get1D_B (host)
			 * Returns the handle to the 1D representation of X.
			 */
			 RawArray get1D_B_Host() { return this->host_b.get(); }

		private:
			uint64_t dim;
			HostArray host_a;
			HostArray host_x;
			HostArray host_b;
			std::vector<RowPtr> rowlist;
			RandomGen<DataType> lsr;
	};

}

#endif
