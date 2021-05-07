#include <algorithm>
#include <stdio.h>
#include <stdlib.h>
#include <cstdlib>
#include <chrono>
#include <iostream>
#include <random>
#include <vector>
#include <cmath>
#include "jacobi_omp.h"
#include "omp.h"


// Multiply A by the x calculated from Jacobi algorithm (used for verification)
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

// Used to verify that array B = test
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

// Prints arr of size n
void print_arr(float* arr, int n)
{
    for (int i = 0; i < n; i++) printf("%f ", arr[i]);
    printf("\n");
}

// Omp implementation of Jacobi Algorithm
void jacobi_omp(float *A, float *B, float *x, int n)
{
    const int ITERATION_LIMIT = 200;
    float LU_sum = 0;
    float * x_new = new float[n];

    for (int k = 0; k < ITERATION_LIMIT; k++)
    {
        #pragma omp for
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
    int n = 1024;
    omp_set_num_threads(2);

    using namespace std::chrono;
    high_resolution_clock::time_point start;
    high_resolution_clock::time_point end;
    duration<double, std::milli> duration_sec;
    
    float* x = new float[n];
    float* A = new float[n*n];
    float* B = new float[n];
    float* test = new float[n];

	std::random_device entropy_source;
	std::mt19937_64 generator(entropy_source());
	std::uniform_real_distribution<float> dist(-1.0,1.0);


	for (int i = 0; i < n; i++)
	{
        for (int j = 0; j < n; j++)
        {
            A[i*n + j] = (float)dist(generator);
        }
        A[i*n + i] = (float)dist(generator) + 2.1; //Diagonally dominant matrix
	}

    for (int i = 0; i < n; i++)
    {
        B[i] = (float)dist(generator);
        x[i] = 1;
    }
/*
    float B_test[4] = {11, 13};
    float A_test[16] = {2, 1, 5, 7};
    float x_test[4] = {1, 1};
*/
    start = high_resolution_clock::now();
    jacobi_omp(A, B, x, n);
    end = high_resolution_clock::now();

    duration_sec = std::chrono::duration_cast<duration<double, std::milli>>(end - start);
   
    printf("ms: %f \n", duration_sec.count());
    // printf("Result: ");
    // print_arr(x,n);

    matmul(A, x, test, n);
    verify(B, test, n);


    delete []A;
    delete []B;
    delete []x;
    delete []test;

    return 0;
}