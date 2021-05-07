#ifndef GMRES_H
#define GMRES_H

// A is n x n matrix in row major order
// b, x are Rn vectors
// e is R{m+1} vector
void gmres(const float *A, const float *b, const float *x, size_t n, float *e,
           const int max_iterations, const float threshold);

#endif