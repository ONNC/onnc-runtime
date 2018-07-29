#include <operator/randomuniform.h>

#include <stdint.h>
#include <stdbool.h>

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
    if(!seed) seed = time(NULL);
    srand(seed);
    float range = high - low;

    int32_t size = 1;
    for(int dim = 0 ; dim < output_output_ndim ; dim++) size *= output_output_dims[dim];

    float rand_number;
    for(int32_t in = 0 ; in < size ; ++in ){
        rand_number = ((float)rand() / (float)RAND_MAX) * range + low;
        output_output[in] = rand_number;
    }
}
