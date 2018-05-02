#pragma once

#include "onnc-runtime.h"

#include <stddef.h>

typedef struct ONNC_RUNTIME_Context {
  const char *weight_file_name;
  int weight_fd; /* Deprecated */
  size_t weight_file_size; /* Deprecated */
  void *weight_mmap_addr; /* Deprecated */
} Context;

typedef struct ONNC_RUNTIME_Tensor_offset TensorOffset;
typedef struct ONNC_RUNTIME_Tensor_offset_table TensorOffsetTable;

