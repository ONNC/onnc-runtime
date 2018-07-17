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
void ONNC_RUNTIME_conv_2d_float(void * onnc_runtime_context,
                                int32_t N, int32_t C, int32_t iH, int32_t iW,
                                const float * X,
                                int32_t M, int32_t kC, int32_t kH, int32_t kW,
                                const float * W,
                                const float * B,
                                int32_t oN, int32_t oC, int32_t oH, int32_t oW,
                                const float * Y,
                                int32_t auto_pad,
                                const int32_t * dilations,
                                int32_t group,
                                const int32_t * kernel_shape,
                                const int32_t * pads,
                                const int32_t * strides);
#else
void ONNC_RUNTIME_conv_2d_float(void *  onnc_runtime_context,
                                int32_t N, int32_t C, int32_t iH, int32_t iW,
                                const float X[ N][C][iH][iW],
                                int32_t M, int32_t kC, int32_t kH, int32_t kW,
                                const float W[ M][kC][kH][kW],
                                const float B[ M],
                                int32_t oN, int32_t oC, int32_t oH, int32_t oW,
                                float Y[ oN][oC][oH][oW],
                                int32_t auto_pad,
                                const int32_t *  dilations,
                                int32_t group,
                                const int32_t *  kernel_shape,
                                const int32_t *  pads,
                                const int32_t *  strides);
#endif

void ONNC_RUNTIME_conv_float(void * onnc_runtime_context,
                             const float * X, const float * W,
                             int32_t ndim, const int32_t * X_dim,
                             const int32_t * W_dim,
                             const float * B, float * Y,
                             const int32_t * Y_dim,
                             int32_t auto_pad,
                             const int32_t * dilations,
                             int32_t group,
                             const int32_t * kernel_shape,
                             const int32_t * pads,
                             const int32_t * strides);

void ONNC_RUNTIME_gemm_float(void * onnc_runtime_context,
                             const float * A,
                             const float * B,
                             int32_t M, int32_t K, int32_t N,
                             const float * C,
                             int32_t ncdim, const int32_t * C_dim,
                             float * Y,
                             int32_t nydim, const int32_t *  Y_dim,
                             float alpha,
                             float beta,
                             int32_t broadcast,
                             int32_t transA,
                             int32_t transB);

void ONNC_RUNTIME_maxpool_float(void *  onnc_runtime_context,
                                const float *  X,
                                int32_t ndim, const int32_t *  X_dim,
                                float *  Y,
                                const int32_t *  Y_dim,
                                int32_t auto_pad,
                                const int32_t *  kernel_shape,
                                const int32_t *  pads,
                                const int32_t *  strides);

void ONNC_RUNTIME_relu_float(void *  onnc_runtime_context,
                             const float *  X,
                             int32_t ndim, const int32_t *  X_dim,
                             float *  Y);

void ONNC_RUNTIME_softmax_float(void *  onnc_runtime_context,
                                const float *  input,
                                int32_t ndim, const int32_t *  input_dim,
                                int32_t axis,
                                float *  output);

void ONNC_RUNTIME_reshape_float(void *  onnc_runtime_context,
                                const float *  data,
                                int32_t ndim, const int32_t *  X_dim,
                                float *  reshaped);

void ONNC_RUNTIME_lrn_float(void *  onnc_runtime_context,
                            const float *  X,
                            int32_t ndim, const int32_t *  X_dim,
                            float alpha,
                            float beta,
                            float bias,
                            int32_t size,
                            float *  Y);

void ONNC_RUNTIME_add_float(void *  onnc_runtime_context,
                            const float *  A,
                            int32_t ndim, const int32_t *  A_dim,
                            const float *  B,
                            float *  C);

void ONNC_RUNTIME_averagepool_float(void *  onnc_runtime_context,
                                    const float *  X,
                                    int32_t ndim, const int32_t *  X_dim,
                                    float *  Y,
                                    const int32_t *  Y_dim,
                                    int32_t auto_pad,
                                    int32_t count_include_pad,
                                    const int32_t *  kernel_shape,
                                    const int32_t *  pads,
                                    const int32_t *  strides);

void ONNC_RUNTIME_batchnormalization_float(void *  onnc_runtime_context,
                                    const float *  X,
                                    int32_t ndim, const int32_t *  X_dim,
                                    float *  Y,
                                    const float *  scale,
                                    const float *  B,
                                    const float *  meanI,
                                    const float *  varI,
                                    float *  meanO,
                                    float *  varO,
                                    float *  saved_mean,
                                    float *  saved_var,
                                    float epsilon,
                                    float momentum,
                                    int32_t spatial);