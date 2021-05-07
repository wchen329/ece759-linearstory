#include "helper.h"

void vec_sub(const float *x, const float *y, float *z, size_t n) {
  for (int i = 0; i < n; i++) {
    z[i] = x[i] - y[i];
  }
}

float dot_product(const float *x, const float *y, size_t n) {
  float ans = 0;
  for (int i = 0; i < n; i++) {
    ans += x[i] * y[i];
  }
  return ans;
}

// assumes A is in row major order!
void mat_vec_mul(const float *A, const float *x, float *y, size_t n) {
  for (int i = 0; i < n; i++) {
    y[i] = dot_product(A + i * n, x, n);
  }
}

float norm(const float *x, size_t n) {
  float total = 0;
  for (int i = 0; i < n; i++) {
    total += x[i] * x[i];
  }
  return sqrt(total);
}

// y = cx
void scalar_mul(const float *x, const float c, float *y, size_t n) {
  for (int i = 0; i < n; i++) {
    y[i] = c * x[i];
  }
}
