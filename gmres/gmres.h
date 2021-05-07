#ifndef GMRES_H
#define GMRES_H

void gmres(const float *A, const float *b, const float *x, size_t n, float *e,
           const int max_iterations, const float threshold);

#endif