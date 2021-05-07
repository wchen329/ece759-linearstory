#include "gmres.h"
#include "helper.h"

void arnoldi(const float *A, float *Q, float *H, size_t n, const int iteration,
             const int max_iterations) {
  float *q = Q + n * iteration;
  float *h = H + (max_iterations + 1) * (iteration - 1);
  mat_vec_mul(A, Q + n * (iteration - 1), q, n);
  float *temp = new float[n];
  for (int i = 0; i < iteration; i++) {
    h[i] = dot_product(q, Q + i * n, n);
    scalar_mul(Q + i * n, h[i], temp, n);
    vec_sub(q, temp, q, n);
  }
  h[iteration] = norm(q, n);
  scalar_mul(q, 1 / h[iteration], q, n);
  delete[] temp;
}

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

  // Q is n x (m + 1) matrix in column major order
  float *Q = new float[n * (max_iterations + 1)];

  for (int i = 0; i < n; i++) {
    Q[i] = r[i] / r_norm;
  }

  float *beta = new float[max_iterations + 1];
  scalar_mul(e1, r_norm, beta, max_iterations + 1);

  // H is (m + 1) x m matrix in column major order
  float *H = new float[(max_iterations + 1) * max_iterations];

  for (int iteration = 0; iteration < max_iterations; iteration++) {
  }
}
