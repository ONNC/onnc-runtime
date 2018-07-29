#include <operator/randomnormal.h>

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>
#include <math.h>

static float randomNormal(float a, float b, float seed);

void ONNC_RUNTIME_randomnormal_float(
  void * restrict onnc_runtime_context  
  ,float * restrict output_output
  ,int32_t output_output_ndim, const int32_t * restrict output_output_dims
  ,int32_t dtype
  ,float mean
  ,float scale
  ,float seed
  ,int32_t * restrict shape
  ,int32_t number_of_shape
) {
  number_of_shape = output_output_ndim;

  int32_t dataSize = 1;
  for(int32_t i = 0; i < output_output_ndim; i++){
    shape[i] = output_output_dims[i];
    dataSize *= output_output_dims[i];
  }

  for(int32_t i = 0; i < dataSize; i++){
    output_output[i] = randomNormal(mean, scale, seed);
  }
}

static float randomNormal(float mean, float stddev, float seed)
{
  srand(seed);
  double pi = acos(-1);
  float x = (float)random() / RAND_MAX + 1;
  float y = (float)random() / RAND_MAX + 1;
  float z = sqrt(-2 * log(x)) * cos(2 * pi * y);
  z = mean + stddev *  z;
  
  return z;
}
