#include <cstdio>
#include <cstdint>

#include "linsys.cuh"

#include "lu_solve_cpu.h"

/* LU Decomposition, sequential
 * Perform an LU Decomposition on some linear system to solve it.
 */
int main(int argc, char ** argv)
{
	// Argument check
	if(argc < 2)
	{
		fprintf(stdout, "Usage: lu_easy n (n is a dim)");
	}

	unsigned long long n = strtoull(argv[1], nullptr, 10);

	linearstory::LUSystem_CPU<int> sys(n);
	sys.solve();
	sys.verify();
}
