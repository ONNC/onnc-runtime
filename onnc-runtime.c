#include "onnc-runtime-internal.h"

#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#include <unistd.h>
#include <sys/stat.h> 
#include <fcntl.h>
#include <sys/mman.h>

void *ONNC_RUNTIME_init_runtime(const char *onnx_weight_file_name) {
  Context *context = (Context *)calloc(1 , sizeof(Context));
  context->weight_file_name = strdup(onnx_weight_file_name);
  context->weight_fd = open(context->weight_file_name, O_RDONLY);
  // XXX: design!
  context->mem = (void **)calloc(2048 , sizeof(void *));
  context->mem_i = 0;

  if (context->weight_fd == -1) {
    // FIXME: handle error
    return NULL;
  }

  // To obtain file size
  struct stat sb;
  if (fstat(context->weight_fd, &sb) == -1) {
    // FIXME: handle error
    return NULL;
  }
  context->weight_file_size = sb.st_size;

  context->weight_mmap_addr =
      mmap(NULL, context->weight_file_size, PROT_READ, MAP_PRIVATE,
           context->weight_fd, 0);
  if (context->weight_mmap_addr == MAP_FAILED) {
    // FIXME: handle error
    return NULL;
  }

  // TODO: Check file magic.

  return context;
}

bool ONNC_RUNTIME_shutdown_runtime(void *onnc_runtime_context) {
  Context *context = (Context *)onnc_runtime_context;
  for (size_t i = 0; i < context->mem_i; ++i) {
    free(context->mem[i]);
  }

  int r = munmap(context->weight_mmap_addr, context->weight_file_size);
  if (r == -1) {
    // FIXME: handle error
    return false;
  }

  r = close(context->weight_fd);
  if (r == -1) {
    // FIXME: handle error
    return false;
  }

  printf("%s\n", context->weight_file_name);
  printf("weight_mmap_addr: %p\n", context->weight_mmap_addr);
  printf("weight_file_size: %zu\n", context->weight_file_size);
  return true;
}

void *ONNC_RUNTIME_load_weight(void *onnc_runtime_context, uint32_t weight_index) {
  Context *context = (Context *)onnc_runtime_context;

  TensorOffsetTable *ttable = (TensorOffsetTable *)context->weight_mmap_addr;
  if (weight_index >= ttable->number_of_tensors) {
    // TODO: Log error
    return NULL;
  }

  return context->weight_mmap_addr + ttable->tensor_offsets[weight_index].offset;
}

void *ONNC_RUNTIME_internal_allocate_memory(void *onnc_runtime_context, size_t num, size_t size) {
  Context *context = (Context *)onnc_runtime_context;
  void *mem = calloc(num , size);
  context->mem[context->mem_i] = mem;
  context->mem_i += 1;
  return mem;
}

int main(int argc, const char *argv[]) {
  if (argc < 2) {
    printf("%s ONNC_WEIGHT\n", argv[0]);
    return EXIT_SUCCESS;
  }
  void *context = ONNC_RUNTIME_init_runtime(argv[1]);
  if (ONNC_RUNTIME_shutdown_runtime(context)) {
    return EXIT_SUCCESS;
  } else {
    return EXIT_FAILURE;
  }
}
