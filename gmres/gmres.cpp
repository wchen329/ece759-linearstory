#include "gmres.h"
#include "helper.h"

void gmres(const float *A, const float *b, const float *x, size_t n, float *e,
           const int max_iterations, const float threshold) {
  float *temp = new float[n];

  mat_vec_mul(A, x, temp, n);
  float *r = new float[n];
  vec_sub(b, temp, r, n);

  float r_norm = norm(r, n);
  float b_norm = norm(b, n);
  float error = r_norm / b_norm;

  float *sn = new float[max_iterations];
  float *cs = new float[max_iterations];
  float *e1 = new float[max_iterations + 1];

  e1[0] = 1;
  for (int i = 0; i < max_iterations; i++) {
    sn[i] = 0;
    cs[i] = 0;
    e1[i + 1] = 0;
  }
  e[0] = error;
}
