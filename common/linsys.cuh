#ifndef __LINSYS_CUH__
#define __LINSYS_CUH__
#include <cstdint>
#include <memory>
#include <random>

#include "cuda_mock.h"

namespace linearstory
{

	/* Linear System: Base Class
	 * A linear system is simply a matrix of several linear functions' coefficients
	 * Several operations can be done over it
	 * Indexed in row major order (though internally, it can be organized differently)
	 *
	 * NOTE: This type can not be copied, as it contains
	 * unique reference to data types
	 */

	typedef uint32_t Coeff;

	__host__ template<class DataType> class LinearSystem
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
			RowPtr operator[](uint32_t ind);
			const RowPtr operator[](uint32_t ind) const;

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
			LinearSystem();
	
		protected:
			/* RawArray get1DData (host)
			 * Returns the handle to the 1D representation of data.
			 */
			 RawArray get1DData_Host() { return this->host_data.get(); }

		private:
			HostArray host_data;
			std::vector<RowPtr> rowlist;
	};

}

#endif
