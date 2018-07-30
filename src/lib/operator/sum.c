#include <operator/sum.h>

#include <stdint.h>
#include <stdbool.h>

void ONNC_RUNTIME_sum_float(
  void * restrict onnc_runtime_context
  ,const float * const * restrict input_data_0
  ,int32_t input_data_0_ntensor
  ,const int32_t * input_data_0_ndim, const int32_t * const * restrict input_data_0_dims
  ,float * restrict output_sum
  ,int32_t output_sum_ndim, const int32_t * restrict output_sum_dims
  
) {
	int32_t size = 1;
	for(int32_t i = 0 ; i < input_data_0_ndim[0] ; ++i){
		size *= input_data_0_dims[0][i];
	}
	for(int32_t i = 0 ; i < size ; ++i){
		output_sum[i] = 0;
		for(int32_t j = 0 ; j < input_data_0_ntensor ; ++j){
			output_sum[i] += input_data_0[j][i] ;
		}
	}
}
