#include <cstdio>
#include <cstdint>
#include <cstdlib>

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
		fprintf(stdout, "Usage: lu_easy n (n is a dim)\n");
		exit(-1);
	}

	unsigned long n = atoi(argv[1]);

	if(n == 0)
	{
		fprintf(stderr, "dim must be > 0\n");
		return -2;
	}

	linearstory::LUSystem_CPU<int> sys(n);
	sys.solve();
	sys.verify();
}
