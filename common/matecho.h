#ifndef __MATECHO_H__
#define __MATECHO_H__
#include <iostream>

namespace linearstory
{
	template<class DataType>
	void MatEcho(DataType* mat_in, size_t n, size_t m)
	{
		// Matrix details
		std::cout << "Matrix" << std::endl
		<< "Rows: " << n << " | Cols: " << m << std::endl;

		// Print row by row
		for(size_t i = 0; i < n; ++i)
		{
			for(size_t j = 0; j < m; ++j)
			{
				std::cout << mat_in[i * m + j] << "\t";
			}

			std::cout << std::endl;
		}
	}
}

#endif

