#include <cstdio>
#include <math.h>
#include <iostream>
#include <cuda.h>
#include "jacobi_kernel.cuh"
#include <random>
#include <vector>

void matmul(float* A, float* x, float* B_test, int n)
{
    for(int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
        {
            B_test[i] += A[i*n + j]*x[j];
        }
    }
    return;
}


void verify(float* B, float* test, int n)
{
    bool flag = true;
    for (int i = 0; i < n; i++)
    {
        if (abs(B[i] - test[i]) > 0.005)
        {
            flag = false;
            printf("%f != %f", B[i], test[i]);
            printf("\n");
        }
        else 
        {
            // printf("%f ", B[i]);
        }
    }
    if(flag == true) printf("SUCCESS");
    else printf("FAILURE");
    return;
}

int main(int argc, char **argv)
{
    int n = 1024;
    int threads_per_block = 32;

	float* hx = new float[n];
	float* hA = new float[n*n];
	float* hB = new float[n];
	float* test = new float[n];

	std::random_device entropy_source;
	std::mt19937_64 generator(entropy_source());
	std::uniform_real_distribution<float> dist(-1.0,1.0);

	for (int i = 0; i < n; i++)
	{
        for (int j = 0; j < n; j++)
        {
            hA[i*n + j] = (float)dist(generator);
        }
        hA[i*n + i] = (float)dist(generator) + 2.1; //Diagonally dominant matrix
	}

    for (int i = 0; i < n; i++)
    {
        hB[i] = (float)dist(generator);
        hx[i] = 1;
    }

	/*
    float hB_test[4] = {6, 25, -11, 15};
    float hA_test[16] = {10, -1, 2, 0, -1 ,11, -1 ,3 ,2 ,-1 ,10, -1, 0, 3, -1, 8};
    float hx_test[4] = {1, 1, 1, 1};
	*/

	jacobi(hA, hB, hx, n, threads_per_block);
	matmul(hA, hx, test, n);
    verify(hB, test, n);



	delete []hA;
	delete []hB;
	delete []hx;
	delete []test;

	return 0;
}
