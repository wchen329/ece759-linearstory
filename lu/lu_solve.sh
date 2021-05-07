#!/usr/bin/env bash
#SBATCH -p wacc
#SBATCH -c 2
#SBATCH -J lu_solve
#SBATCH -o lu_solve.out -e lu_solve.err
make clean
make
./lu_easy.seq 512
./lu_par.omp 512