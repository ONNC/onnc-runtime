#pragma once

#include <stdint.h>
#include <stdbool.h>

struct ONNC_RUNTIME_Tensor_offset {
  uint64_t offset; /* Tensor offset */
  uint64_t size;   /* Size of tensor in bytes */
};

#define ONNC_RUNTIME_TENSOR_FILE_MAGIC ".TSR"

struct ONNC_RUNTIME_Tensor_offset_table {
  uint8_t magic[4];                                    /* Tensor File magic number. */
  uint32_t number_of_tensors;
  struct ONNC_RUNTIME_Tensor_offset tensor_offsets[];
};

/**
 * Initialize runtime by onnx model file.
 * @deprecated
 * @param onnx_model_file_name onnx model file name with path.
 * @return The ONNC Runtime Context, should be passed to every ONNC Runtime functions.
 */
void *ONNC_RUNTIME_init_runtime(const char *onnx_model_file_name);

/**
 * Shutdown runtime.
 * @deprecated
 * @param onnc_runtime_context The ONNC Runtime Context.
 * @return True if shutdown successfully. False if something wrong.
 */
bool ONNC_RUNTIME_shutdown_runtime(void *onnc_runtime_context);

/**
 * Load weight to memory.
 * @param onnc_runtime_context the ONNC Runtime Context.
 * @param weight_index the weight index.
 * @return The memory which contains the weight[weight_index].
 */
void *ONNC_RUNTIME_load_weight(void *onnc_runtime_context, uint32_t weight_index);

void ONNC_RUNTIME_conv_float(void * restrict onnc_runtime_context,
                             const float * restrict X, const float * restrict W,
                             int32_t ndim, const int32_t * restrict X_dim,
                             const int32_t * restrict W_dim,
                             const float * restrict B, float * restrict Y,
                             const int32_t * restrict Y_dim,
                             int32_t auto_pad,
                             const int32_t * restrict dilations,
                             int32_t group,
                             const int32_t * restrict kernel_shape,
                             const int32_t * restrict pads,
                             const int32_t * restrict strides);

void ONNC_RUNTIME_gemm_float(void * restrict onnc_runtime_context,
                             const float * restrict A,
                             const float * restrict B,
                             const float * restrict C,
                             int32_t M, int32_t K, int32_t N,
                             float * restrict Y,
                             int32_t ndim, const int32_t * restrict Y_dim,
                             float alpha,
                             float beta,
                             int32_t broadcast,
                             int32_t transA,
                             int32_t transB);

void ONNC_RUNTIME_maxpool_float(void * restrict onnc_runtime_context,
                                const float * restrict X,
                                int32_t ndim, const int32_t * restrict X_dim,
                                float * restrict Y,
                                const int32_t * restrict Y_dim,
                                int32_t auto_pad,
                                const int32_t * restrict kernel_shape,
                                const int32_t * restrict pads,
                                const int32_t * restrict strides);

void ONNC_RUNTIME_relu_float(void * restrict onnc_runtime_context,
                             const float * restrict X,
                             int32_t ndim, const int32_t * restrict X_dim,
                             float * restrict Y);

void ONNC_RUNTIME_softmax_float(void * restrict onnc_runtime_context,
                                const float * restrict X,
                                int32_t ndim, const int32_t * restrict X_dim,
                                int32_t axis,
                                float * restrict Y);

void ONNC_RUNTIME_reshape_float(void * restrict onnc_runtime_context,
                                const float * restrict data,
                                int32_t ndim, const int32_t * restrict X_dim,
                                float * restrict reshaped);

void ONNC_RUNTIME_LRN_float(void * restrict onnc_runtime_context,
                            const float * restrict X,
                            int32_t ndim, const int32_t * restrict X_dim,
                            float alpha,
                            float beta,
                            float bias,
                            int32_t size,
                            float * restrict Y);
