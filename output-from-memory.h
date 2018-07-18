#pragma once

#include <stdbool.h>
#include <stdint.h>

bool ONNC_RUNTIME_initialize_output_memory(void *onnc_runtime_context,
                                           void *addr);
bool ONNC_RUNTIME_finalize_output_memory(void *context);

void *ONNC_RUNTIME_get_output_memory(void *onnc_runtime_context, uint32_t output_index);
