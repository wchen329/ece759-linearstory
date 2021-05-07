#ifndef HELPER_H
#define HELPER_H

#include <cmath>

// z = x - y
void vec_sub(const float *x, const float *y, float *z, size_t n);

float dot_product(const float *x, const float *y, size_t n);

// y = Ax
void mat_vec_mul(const float *A, const float *x, float *y, size_t n);

float norm(const float *x, size_t n);

// y = cx
void scalar_mul(const float *x, const float c, float *y, size_t n);

#endif