#pragma once

#include <operator/clip.h>

#include <stdint.h>
#include <stdbool.h>

void ONNC_RUNTIME_clip_float(
  void * restrict onnc_runtime_context
  ,const float * restrict input_input
  ,int32_t input_input_ndim, const int32_t * restrict input_input_dims
  ,float * restrict output_output
  ,int32_t output_output_ndim, const int32_t * restrict output_output_dims
  ,float max
  ,float min
) {

	int32_t size = 1;
	int32_t numofdim = input_input_ndim;

	for(int32_t i = 0 ; i < numofdim ; ++i){
		size *= input_input_dims[i];
	}

	for(int32_t i = 0 ; i < size ; ++i){
	    float tmp_val = input_input[i];
	    tmp_val = tmp_val > max ? max : tmp_val;
	    tmp_val = tmp_val < min ? min : tmp_val;
		output_output[i] = tmp_val;
	}
}

