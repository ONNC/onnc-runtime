#pragma once

#include <stdint.h>
#include <stdbool.h>

struct ONNC_RUNTIME_Tensor_offset {
  uint64_t offset; /* Tensor offset */
  uint64_t size;   /* Size of tensor in bytes */
};

#define ONNC_RUNTIME_TENSOR_FILE_MAGIC ".TSR"

struct ONNC_RUNTIME_Tensor_offset_table {
  uint8_t magic[8];                                    /* Tensor File magic number. */
  uint64_t number_of_tensors;
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

#ifdef __cplusplus
void ONNC_RUNTIME_conv_2d_float(void * restrict onnc_runtime_context,
                                int32_t N, int32_t C, int32_t iH, int32_t iW,
                                const float * restrict X,
                                int32_t M, int32_t kC, int32_t kH, int32_t kW,
                                const float * restrict W,
                                const float * restrict B,
                                int32_t oN, int32_t oC, int32_t oH, int32_t oW,
                                const float * restrict Y,
                                int32_t auto_pad,
                                const int32_t * restrict dilations,
                                int32_t group,
                                const int32_t * restrict kernel_shape,
                                const int32_t * restrict pads,
                                const int32_t * restrict strides);
#else
void ONNC_RUNTIME_conv_2d_float(void * restrict onnc_runtime_context,
                                int32_t N, int32_t C, int32_t iH, int32_t iW,
                                const float X[restrict N][C][iH][iW],
                                int32_t M, int32_t kC, int32_t kH, int32_t kW,
                                const float W[restrict M][kC][kH][kW],
                                const float B[restrict M],
                                int32_t oN, int32_t oC, int32_t oH, int32_t oW,
                                float Y[restrict oN][oC][oH][oW],
                                int32_t auto_pad,
                                const int32_t * restrict dilations,
                                int32_t group,
                                const int32_t * restrict kernel_shape,
                                const int32_t * restrict pads,
                                const int32_t * restrict strides);
#endif

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
                             int32_t M, int32_t K, int32_t N,
                             const float * restrict C,
                             int32_t ncdim, const int32_t * restrict C_dim,
                             float * restrict Y,
                             int32_t nydim, const int32_t * restrict Y_dim,
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
                                const float * restrict input,
                                int32_t ndim, const int32_t * restrict input_dim,
                                int32_t axis,
                                float * restrict output);

void ONNC_RUNTIME_reshape_float(void * restrict onnc_runtime_context,
                                const float * restrict data,
                                int32_t ndim, const int32_t * restrict X_dim,
                                float * restrict reshaped);

void ONNC_RUNTIME_lrn_float(void * restrict onnc_runtime_context,
                            const float * restrict X,
                            int32_t ndim, const int32_t * restrict X_dim,
                            float alpha,
                            float beta,
                            float bias,
                            int32_t size,
                            float * restrict Y);

void ONNC_RUNTIME_add_float(void * restrict onnc_runtime_context,
                            const float * restrict A,
                            int32_t ndim, const int32_t * restrict A_dim,
                            const float * restrict B,
                            float * restrict C);

void ONNC_RUNTIME_averagepool_float(void * restrict onnc_runtime_context,
                                    const float * restrict X,
                                    int32_t ndim, const int32_t * restrict X_dim,
                                    float * restrict Y,
                                    const int32_t * restrict Y_dim,
                                    int32_t auto_pad,
                                    int32_t count_include_pad,
                                    const int32_t * restrict kernel_shape,
                                    const int32_t * restrict pads,
                                    const int32_t * restrict strides);
