#include <algorithm>
#include<stdio.h>
#include<stdlib.h>
#include<cstdlib>
#include<chrono>
#include <iostream>
#include <random>
#include <vector>

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
    print_arr(x, n);

    for (int k = 0; k < ITERATION_LIMIT; k++)
    {
        for (int i = 0 ; i < n; i++)
        {
            LU_sum = 0;
            for (int j = 0; j < n; j++)
            {
                if (j != i)
                {
                    LU_sum += A[i*n + j]*x[j];
                }
            }
            x_new[i] = B[i] - LU_sum;
            x_new[i] = x_new[i]/A[i*n + i];
        }
        // x = x_new;
        std::copy(x_new, x_new + n, x);
        print_arr(x, n);

    }
    delete[] x_new;
}

int main(int argc, char **argv)
{
    int n = 4;

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

    B[0] = 6;
    B[1] = 25;
    B[2] = -11;
    B[3] = 15;
    
    A[0] = 10;
    A[1] = -1;
    A[2] = 2;
    A[3] = 0;
    A[4] = -1;
    A[5] = 11;
    A[6] = -1;
    A[7] = 3;
    A[8] = 2;
    A[9] = -1;
    A[10] = 10;
    A[11] = -1;
    A[12] = 0;
    A[13] = 3;
    A[14] = -1;
    A[15] = 8;
    
    x[0] = 1;
    x[1] = 1;
    x[2] = 1;
    x[3] = 1;

    start = high_resolution_clock::now();
    jacobi(A,B,x,n);
    end = high_resolution_clock::now();

    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
   

    printf("ms: %f \n", duration_sec.count());
    printf("Result: \n");
    print_arr(x,n);

    delete []A;
    delete []B;
    delete []x;

    return 0;
}
