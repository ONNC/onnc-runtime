#include <operator/xor.h>

#include <stdint.h>
#include <stdbool.h>

//function declaration
static void tile_float(
  const float * restrict input_input
  ,int32_t input_input_ndim, const int32_t * restrict input_input_dims
  ,const float * restrict input_repeats
  ,float * restrict output_output
  ,int32_t output_output_ndim, const int32_t * restrict output_output_dims
);

void ONNC_RUNTIME_xor_float(
  void * restrict onnc_runtime_context
  ,const float * restrict input_A
  ,int32_t input_A_ndim, const int32_t * restrict input_A_dims
  ,const float * restrict input_B
  ,int32_t input_B_ndim, const int32_t * restrict input_B_dims
  ,float * restrict output_C
  ,int32_t output_C_ndim, const int32_t * restrict output_C_dims
  
) {
  
}
