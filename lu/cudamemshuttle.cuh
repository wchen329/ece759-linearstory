#ifndef __CUDAMEMSHUTTLE_CUH__
#define __CUDAMEMSHUTTLE_CUH__
#include <cuda_runtime.h>
#include <cuda_runtime_api.h>

namespace linearstory
{
	template<class DataType>
	class CudaMemoryShuttle
	{
		public:
			/* constructor
			 * Make a CUDA memory zone of size n
			 */
			CudaMemoryShuttle(size_t n) :
				arr_count(n)
			{
				// Make my memory
				cudaMalloc(&device_arr, n * sizeof(DataType));
			}

			/* destructor
			 * Free up the memory region.
			 */
			~CudaMemoryShuttle()
			{
				cudaFree(device_arr);
			}

			/* pushHostArr
			 * Push the host array to the device array
			 *
			 */
			void pushHostArr(DataType* hr)
			{
				cudaMemcpy(device_arr, hr, arr_count * sizeof(DataType), cudaMemcpyHostToDevice);
			}

			/* pullHostArr
			 * Pull the host array back from device
			 */
			void pullHostArr(DataType* hr)
			{
				cudaMemcpy(hr, device_arr, arr_count * sizeof(DataType), cudaMemcpyDeviceToHost);
			}

			/* zero
			 * Zero array on device
			 */
			void zero()
			{
				cudaMemset(device_arr, 0, arr_count * sizeof(DataType));
			}

			/* raw
			 * Get raw device pointer
			 */
			DataType* raw()
			{
				return device_arr;
			}

		private:
			DataType* device_arr;
			size_t arr_count;

			// This is unique
			CudaMemoryShuttle operator=(CudaMemoryShuttle&);
			CudaMemoryShuttle(CudaMemoryShuttle&);
	};
}

#endif // ! __CUDAMEMSHUTTLE_CUH__
