#pragma once

#include <stdbool.h>
#include <stdint.h>

bool ONNC_RUNTIME_initialize_input_memory(void *context, void *addr);
bool ONNC_RUNTIME_finalize_input_memory(void *context);

void *ONNC_RUNTIME_get_input_memory(void *onnc_runtime_context, uint32_t input_index);
