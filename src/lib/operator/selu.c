#include <operator/selu.h>

#include <stdint.h>
#include <stdbool.h>
#include <math.h>

void ONNC_RUNTIME_selu_float(
  void * restrict onnc_runtime_context
  ,const float * restrict input_X
  ,int32_t input_X_ndim, const int32_t * restrict input_X_dims
  ,float * restrict output_Y
  ,int32_t output_Y_ndim, const int32_t * restrict output_Y_dims
  ,float alpha
  ,float gamma
) {
	int32_t size = 1;

	for(int32_t i = 0 ; i < input_X_ndim ; ++i){
		size *= input_X_dims[i];
	}

    // y = gamma * (alpha * e^x - alpha) for x <= 0
    // y = gamma * x for x > 0

    // X = (X > 0) ? X : alpha *(expf(X) - 1.0f)
    // y = gamma * X;
	for(int32_t i = 0 ; i < size ; ++i){
	    float tmp_val = input_X[i];
        if(tmp_val <= 0.0f) tmp_val = alpha * (expf(tmp_val) - 1.0f);
        output_Y[i] = gamma * tmp_val;
	}
}

