#pragma once

#include <operator/sigmoid.h>

#include <stdint.h>
#include <stdbool.h>
#include <math.h>

void ONNC_RUNTIME_sigmoid_float(
  void * restrict onnc_runtime_context
  ,const float * restrict input_X
  ,int32_t input_X_ndim, const int32_t * restrict input_X_dims
  ,float * restrict output_Y
  ,int32_t output_Y_ndim, const int32_t * restrict output_Y_dims
  
) {
	int32_t size = 1;
	int32_t numofdim = input_X_ndim;

	for(int32_t i = 0 ; i < numofdim ; ++i){
		size *= input_X_dims[i];
	}

    //y = 1 / (1 + exp(-x))
	for(int32_t i = 0 ; i < size ; ++i){
		output_Y[i] = 1.0f / (1.0f + expf(input_X[i] * -1.0f));
	}
}

