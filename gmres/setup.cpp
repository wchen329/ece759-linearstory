#include "setup.h"
#include <cstdlib>
#include <random>

void generate(float *A, float *B, size_t m) {
  std::random_device entropy_source;
  std::default_random_engine generator(entropy_source());
  std::uniform_real_distribution<float> dist(-100.0, 100.0);

  for (size_t i = 0; i < m; i++) {
    for (size_t j = 0; j < m; j++) {
      A[i * m + j] = dist(generator);
    }
  }

  for (size_t i = 0; i < m; i++) {
    B[i] = dist(generator);
  }
}

bool test(const float *A, const float *B, const float *x, size_t m) {
  for (int i = 0; i < m; i++) {
    float bi = 0;
    for (int j = 0; j < m; j++) {
      bi += A[i * m + j] * x[j];
    }
    if (abs(bi - B[i]) > EPSILON) {
      return false;
    }
  }
  return true;
}