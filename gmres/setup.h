#ifndef SETUP_H
#define SETUP_H

const float EPSILON = 1e-3;

void generate(float *A, float *B, size_t m);
bool test(const float *A, const float *B, const float *x, size_t m);

#endif