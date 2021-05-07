#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <chrono>
#include "linsys.cuh"
#include <omp.h>
#include "lu_solve_par.h"

/* LU Decomposition, sequential
 * Perform an LU Decomposition on some linear system to solve it.
 */
int main(int argc, char ** argv)
{
	// Argument check
	if(argc < 2)
	{
		fprintf(stdout, "Usage: lu_par n (n is a dim)\n");
		exit(-1);
	}
	if (argc < 3)
	{
		fprintf(stdout, "Usage: lu_par t unspecified\n");
		exit(-1);
	}

	unsigned long n = atoi(argv[1]);
	int t = atoi(argv[2]);
	omp_set_num_threads(t);

	if(n == 0)
	{
		fprintf(stderr, "dim must be > 0\n");
		return -2;
	}

	using namespace std::chrono;
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;

	linearstory::LUSystem_CPU<float> sys(n);

	start = high_resolution_clock::now();
	#pragma omp parallel
		sys.solve();

	end = high_resolution_clock::now();
    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);

	printf("par ms: %f \n", duration_sec.count());

	// Uncomment to enable verification
	// sys.verify();
}
