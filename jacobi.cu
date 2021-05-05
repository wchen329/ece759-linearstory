#include <cstdio>
#include <math.h>
#include <iostream>
#include <cuda.h>
#include "jacobi_kernel.cuh"
#include <random>
#include <vector>

int main(int argc, char **argv)
{
    /*
	int n = 0;
	int threads_per_block = 0;
	if( argc > 2)
	{
		n = atoi(argv[1]);
		threads_per_block = atoi(argv[2]);
	}
    */
    int n = 4;
    int threads_per_block = 32;

	float* hx = new float[n];
	float* hA = new float[n*n];
	float* hB = new float[n];
/*
	std::random_device entropy_source;
	std::mt19937_64 generator(entropy_source());
	std::uniform_real_distribution<double> dist(-1.0,1.0);
	for (int i = 0; i < n*n; i++)
	{
		hA[i] = (float)dist(generator);
		hB[i] = (float)dist(generator);
	}
*/

    float hB_test[4] = {6, 25, -11, 15};
    float hA_test[16] = {10, -1, 2, 0, -1 ,11, -1 ,3 ,2 ,-1 ,10, -1, 0, 3, -1, 8};
    float hx_test[4] = {1, 1, 1, 1};

	jacobi(hA_test, hB_test, hx_test, n, threads_per_block);

	delete []hA;
	delete []hB;
	delete []hx;

	return 0;
}
