#include <cstdlib>
#include <cstdio>
#include <cstdint>

#include "linsys.cuh"
#include "lu_solve_cuda.cuh"

int main(int argc, char ** argv)
{
	if (argc < 3)
	{
		fprintf(stderr, "Usage: lu_cuda n threads_per_block\n");
		exit(-1);
	}

	unsigned n = atoi(argv[1]);
	unsigned threads_per_block = atoi(argv[2]);

	if(!n || !threads_per_block)
	{
		fprintf(stderr, "n and threads_per_block should be greater than 0");
		exit(-2);
	}

	// Make the system
	linearstory::LU_CUDA_System<float> sys(n, threads_per_block);

	// Solve and verify
	sys.solve();
	bool iscorrect = sys.verify();
	if (!iscorrect)
	{
		fprintf(stdout, "There were errors.\n");
	}
	fflush(stdout);

	return 0;
}