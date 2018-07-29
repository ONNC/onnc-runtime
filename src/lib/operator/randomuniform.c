#include <operator/randomuniform.h>

#include <stdint.h>
#include <stdbool.h>
#include <stdlib.h>

static float uniform(float a, float b);

void ONNC_RUNTIME_randomuniform_float(
  void * restrict onnc_runtime_context
  ,float * restrict output_output
  ,int32_t output_output_ndim, const int32_t * restrict output_output_dims
  ,int32_t dtype
  ,float high
  ,float low
  ,float seed
  ,int32_t * restrict shape
  ,int32_t number_of_shape
) {
  number_of_shape = output_output_ndim;

  int32_t dataSize = 1;
  for(int32_t i =0; i < output_output_ndim; i++){
    shape[i] = output_output_dims[i];
    dataSize *= shape[i];
  }
  srand(seed);
  for(int32_t i =0; i < dataSize; i++){
    output_output[i] = uniform(low, high);
  }
}


float uniform(float low, float high)
{
  return rand() / (RAND_MAX + 1.0) * (high - low) + low;
}