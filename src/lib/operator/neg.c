#include <operator/neg.h>

#include <stdint.h>
#include <stdbool.h>
#include <stdio.h>

void ONNC_RUNTIME_neg_float(
  void * restrict onnc_runtime_context
  ,const float * restrict input_X
  ,int32_t input_X_ndim, const int32_t * restrict input_X_dims
  ,float * restrict output_Y
  ,int32_t output_Y_ndim, const int32_t * restrict output_Y_dims
  
) {
  
  int32_t i;
  int32_t dataSize = 1;
  for (i = 0; i < input_X_ndim; i++){
    dataSize *= input_X_dims[i];
  }
  
  int32_t j;
  for(j = 0; j < dataSize; j++){
    output_Y[j] = -input_X[j];
  }
}
