#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -c 2
#SBATCH -J jacobi_kernel
#SBATCH -o jacobi_kernel.out -e jacobi_kernel.err
#SBATCH --gres=gpu:1 -c 1

module load cuda
nvcc jacobi.cu jacobi_kernel.cu -Xcompiler -O3 -Xcompiler -Wall -Xptxas -O3 -std c++17 -o jacobi_kernel
./jacobi_kernel