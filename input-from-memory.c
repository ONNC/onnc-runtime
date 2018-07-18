#include "input-from-memory.h"
#include "file-context.h"
#include "onnc-runtime-internal.h"

#include <stdlib.h>
#include <string.h>
#include <stdio.h>

typedef struct ONNC_RUNTIME_Input_Memory_Context {
  void *orig_context;
  void *addr;
} InputContext;

bool ONNC_RUNTIME_initialize_input_memory(void *onnc_runtime_context, void *addr) {
  Context *onnc_context = (Context *)onnc_runtime_context;
  // FIXME: check onnc_context

  if (strncmp(addr,
              ONNC_RUNTIME_TENSOR_FILE_MAGIC,
              sizeof(ONNC_RUNTIME_TENSOR_FILE_MAGIC) - 1) != 0) {
    // FIXME: handle error
    fprintf(stderr,
            "Invalid input tensors. Magic does not match: \"%.*s\" \"%.*s\".",
            (int)sizeof(ONNC_RUNTIME_TENSOR_FILE_MAGIC) - 1,
            ONNC_RUNTIME_TENSOR_FILE_MAGIC,
            (int)sizeof(ONNC_RUNTIME_TENSOR_FILE_MAGIC) - 1,
            (char *)addr);
    return false;
  }

  InputContext *context = (InputContext *)calloc(1 , sizeof(InputContext));

  context->addr = addr;
  context->orig_context = onnc_context->input_context;
  onnc_context->input_context = context;
  return true;
}

bool ONNC_RUNTIME_finalize_input_memory(void *onnc_runtime_context) {
  Context *onnc_context = (Context *)onnc_runtime_context;
  InputContext *context = (InputContext *)onnc_context->input_context;
  onnc_context->input_context = context->orig_context;

  free(context);
  return true;
}

void *ONNC_RUNTIME_get_input_memory(void *onnc_runtime_context,
                                    uint32_t index) {
  Context *onnc_context = (Context *)onnc_runtime_context;
  InputContext *context = (InputContext *)onnc_context->input_context;
  return ONNC_RUNTIME_load_from_tensor_table(context->addr, index);
}
