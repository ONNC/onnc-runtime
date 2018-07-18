#include "onnc-runtime-internal.h"

#include "input-from-memory.h"
#include "weight-from-memory.h"
#include "output-from-memory.h"

#include "file-context.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, const char *argv[]) {
  if (argc < 4) {
    printf("%s ONNC_INPUT ONNC_WEIGHT ONNC_OUTPUT\n", argv[0]);
    return EXIT_SUCCESS;
  }
  Context *context = (Context *)ONNC_RUNTIME_init_runtime();
  FileContext *file_context;

  // FIXME: check return value, handle error
  file_context = ONNC_RUNTIME_initialize_file_context_read(argv[2]);
  context->weight_context = file_context;
  ONNC_RUNTIME_initialize_weight_memory(context, file_context->addr);

  // FIXME: check return value, handle error
  file_context = ONNC_RUNTIME_initialize_file_context_read(argv[1]);
  context->input_context = file_context;
  ONNC_RUNTIME_initialize_input_memory(context, file_context->addr);

  context->output_context = (void *)argv[3];

  model_main(context);

  // FIXME: check return value, handle error
  ONNC_RUNTIME_finalize_input_memory(context);
  ONNC_RUNTIME_finalize_file_context(context->input_context);
  // FIXME: check return value, handle error
  ONNC_RUNTIME_finalize_output_memory(context);
  ONNC_RUNTIME_finalize_file_context(context->output_context);

  // FIXME: check return value, handle error
  ONNC_RUNTIME_finalize_weight_memory(context);
  ONNC_RUNTIME_finalize_file_context(context->weight_context);

  if (ONNC_RUNTIME_shutdown_runtime(context)) {
    return EXIT_SUCCESS;
  } else {
    return EXIT_FAILURE;
  }
}

/* TODO: Design input, output, weight, memory module. */
bool ONNC_RUNTIME_initialize_output(void *onnc_runtime_context,
                                    uint32_t num_of_output,
                                    uint32_t *outputs_sizes) {
  uint64_t all_size = 0;
  uint64_t offset = sizeof(TensorOffsetTable);
  for (uint32_t i = 0; i < num_of_output; ++i) {
    all_size += outputs_sizes[i];
    offset += sizeof(TensorOffset);
  }
  all_size += offset;

  Context *onnc_context = (Context *)onnc_runtime_context;

  FileContext *file_context =
    ONNC_RUNTIME_initialize_file_context_write(
      (const char *)onnc_context->output_context,
      all_size
    );
  onnc_context->output_context = file_context;
  void *addr = file_context->addr;

  strncpy(addr,
          ONNC_RUNTIME_TENSOR_FILE_MAGIC,
          sizeof(ONNC_RUNTIME_TENSOR_FILE_MAGIC) - 1);

  TensorOffsetTable *ttable = (TensorOffsetTable *)addr;
  for (uint32_t i = 0; i < num_of_output; ++i) {
    ttable->tensor_offsets[i].offset = offset;
    ttable->tensor_offsets[i].size = outputs_sizes[i];
    offset += outputs_sizes[i];
  }

  // FIXME: check return value, handle error
  ONNC_RUNTIME_initialize_output_memory(onnc_context, addr);
  return true;
}

// XXX: It's a sample.
void model_main(void *context){
  uint32_t sizes[1] = {10};
  ONNC_RUNTIME_initialize_output(context, 1, sizes);

  // node { input: "gpu_0/data_0" input: "gpu_0/conv1_w_0" output: "gpu_0/conv1_1" name: "" op_type: "Conv" attribute { name: "pads" ints: 3 ints: 3 ints: 3 ints: 3 type: INTS } attribute { name: "kernel_shape" ints: 7 ints: 7 type: INTS } attribute { name: "strides" ints: 2 ints: 2 type: INTS } }
  void *data_0;
  void *conv1_w_0;
  data_0 = ONNC_RUNTIME_get_input_memory(context, 0);
  printf("%p\n", data_0);
  conv1_w_0 = ONNC_RUNTIME_get_weight_memory(context, 0);
  printf("%p\n", conv1_w_0);

  //void *conv1_1;
  // TODO: design memory space
  // conv1_1 = ONNC_RUNTIME_get_memory(context, 0);
  // ONNC_RUNTIME_conv_2d_float(data_0, conv1_w_0, conv1_1, ...attributes/shape.....);
}


