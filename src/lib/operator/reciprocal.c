#pragma once

#include <operator/reciprocal.h>

#include <stdint.h>
#include <stdbool.h>
#include <math.h>

void ONNC_RUNTIME_reciprocal_float(
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

    // y = 1/x, x!= 0.0f
    // y = INF, if x = 0.0f    
	for(int32_t i = 0 ; i < size ; ++i){
		float tmp_val = input_X[i];
		output_Y[i] = tmp_val == 0.0f ? INFINITY : 1.0f / tmp_val;
	}
}

