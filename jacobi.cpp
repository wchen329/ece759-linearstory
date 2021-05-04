#include <algorithm>
#include<stdio.h>
#include<stdlib.h>
#include<cstdlib>
#include<chrono>
#include <iostream>
#include <random>
#include <vector>
#include "omp.h"

void print_arr(float* arr, int n)
{
    printf("\n");
    for (int i = 0; i < n; i++)
    {
        printf("%f ", arr[i]);
    }
    printf("\n");
}

void jacobi(float *A, float *B, float *x, int n)
{
    const int ITERATION_LIMIT = 1000;
    float LU_sum = 0;
    float * x_new = new float[n];

    #pragma omp for
        for (int k = 0; k < ITERATION_LIMIT; k++)
        {
            for (int i = 0 ; i < n; i++)
            {
                LU_sum = 0;
                for (int j = 0; j < n; j++)
                {   
                    LU_sum += A[i*n + j]*x[j];
                }
                LU_sum -= A[i*n+i]*x[i];
                x_new[i] = B[i] - LU_sum;
                x_new[i] = x_new[i]/A[i*n + i];
            }
            std::copy(x_new, x_new + n, x);
        }
    delete[] x_new;
}

int main(int argc, char **argv)
{
    int n = 4;
    omp_set_num_threads(2);

    using namespace std::chrono;
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    
    float* x = new float[n];
    float* A = new float[n*n];
    float* B = new float[n];

	std::random_device entropy_source;
	std::mt19937_64 generator(entropy_source());
	std::uniform_real_distribution<float> dist(-1.0,1.0);

	// A[i] = (float)dist(generator);

	for (int i = 0; i < n*n; i++)
	{
        A[i] = 1;
	}

    for (int i = 0; i < n; i++)
    {
        x[i] = 0;
        B[i] = 1;
    }

    float B_test[4] = {6, 25, -11, 15};
    float A_test[16] = {10, -1, 2, 0, -1 ,11, -1 ,3 ,2 ,-1 ,10, -1, 0, 3, -1, 8};
    float x_test[4] = {1, 1, 1, 1};

    start = high_resolution_clock::now();
    jacobi(A_test, B_test, x_test, n);
    end = high_resolution_clock::now();

    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
   

    printf("ms: %f \n", duration_sec.count());
    printf("Result: ");
    print_arr(x_test,n);

    delete []A;
    delete []B;
    delete []x;

    return 0;
}