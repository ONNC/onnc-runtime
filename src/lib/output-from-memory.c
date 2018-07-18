#include <output-from-memory.h>
#include <file-context.h>
#include <onnc-runtime-internal.h>

#include <stdlib.h>
#include <string.h>

typedef struct ONNC_RUNTIME_Output_Memory_Context {
  void *orig_context;
  void *addr;
} OutputContext;

bool ONNC_RUNTIME_initialize_output_memory(void *onnc_runtime_context,
                                           void *addr) {
  Context *onnc_context = (Context *)onnc_runtime_context;
  // FIXME: check onnc_context

  OutputContext *context = (OutputContext *)calloc(1 , sizeof(OutputContext));

  strncpy(addr,
          ONNC_RUNTIME_TENSOR_FILE_MAGIC,
          sizeof(ONNC_RUNTIME_TENSOR_FILE_MAGIC) - 1);

  context->addr = addr;
  context->orig_context = onnc_context->output_context;
  onnc_context->output_context = context;
  return true;
}

bool ONNC_RUNTIME_finalize_output_memory(void *onnc_runtime_context) {
  Context *onnc_context = (Context *)onnc_runtime_context;
  OutputContext *context = (OutputContext *)onnc_context->output_context;
  onnc_context->output_context = context->orig_context;

  free(context);
  return true;
}

void *ONNC_RUNTIME_get_output_memory(void *onnc_runtime_context,
                                     uint32_t index) {
  Context *onnc_context = (Context *)onnc_runtime_context;
  OutputContext *context = (OutputContext *)onnc_context->output_context;
  return ONNC_RUNTIME_load_from_tensor_table(context->addr, index);
}
