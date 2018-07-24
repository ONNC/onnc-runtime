#pragma once

#include <stdint.h>
#include <stdbool.h>

void ONNC_RUNTIME_if_float(
  void * restrict onnc_runtime_context
  ,const float * restrict input_cond
  ,int32_t input_cond_ndim, const int32_t * restrict input_cond_dims
  ,float ** restrict output_outputs
  ,int32_t output_outputs_ntensor
  ,int32_t * output_outputs_ndim, const int32_t ** restrict output_outputs_dims
  ,void * restrict else_branch
  ,void * restrict then_branch
);
