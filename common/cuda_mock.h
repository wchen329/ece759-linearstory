#ifndef __CUDA_MOCK_H__
#define __CUDA_MOCK_H__

	// The following makes some CUDA constructs compilable, but not functional
	// on a regular GCC compiler. It is only activated if specified in compile.
	#ifdef ENABLE_MOCK
		#define __host__ // disable host
		#define __device__  // disable device key

	#endif

#endif
