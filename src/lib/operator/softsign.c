#include <operator/softsign.h>

#include <stdint.h>
#include <stdbool.h>
#include <math.h>

void ONNC_RUNTIME_softsign_float(
  void * restrict onnc_runtime_context
  ,const float * restrict input_input
  ,int32_t input_input_ndim, const int32_t * restrict input_input_dims
  ,float * restrict output_output
  ,int32_t output_output_ndim, const int32_t * restrict output_output_dims
  
) {
	int32_t size = 1;

	for(int32_t i = 0 ; i < input_input_ndim ; ++i){
		size *= input_input_dims[i];
	}

    //y = (x/(1+|x|)) 
	for(int32_t i = 0 ; i < size ; ++i){
	    float tmp_val = input_input[i];
        output_output[i] = tmp_val / (fabsf(tmp_val) + 1.0f);
	}
}

