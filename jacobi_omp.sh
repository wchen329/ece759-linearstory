#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -c 1
#SBATCH -J jacobi_omp
#SBATCH -o jacobi_omp.out -e jacobi_omp.err
module load gcc/10.2.0
g++ jacobi_omp.cpp -Wall -O3 -std=c++17 -o jacobi_omp -fopenmp
./jacobi_omp