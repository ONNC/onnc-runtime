#pragma once

#include "onnc-runtime.h"

#include <stddef.h>

typedef struct ONNC_RUNTIME_Context {
  const char *weight_file_name;
  int weight_fd; /* Deprecated */
  size_t weight_file_size; /* Deprecated */
  void *weight_mmap_addr; /* Deprecated */
  void **mem; /* Deprecated */
  size_t mem_i; /* Deprecated */
} Context;


void *ONNC_RUNTIME_internal_allocate_memory(void *onnc_runtime_context, size_t num, size_t size);

typedef struct ONNC_RUNTIME_Tensor_offset TensorOffset;
typedef struct ONNC_RUNTIME_Tensor_offset_table TensorOffsetTable;

