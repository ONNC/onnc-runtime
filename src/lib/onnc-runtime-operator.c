#include <onnc-runtime-internal.h>

#include <assert.h>
#include <string.h>
#include <float.h>
#include <math.h>

#include <stdio.h>
#include <stdlib.h>
#include <inttypes.h>

static inline bool next_dim(int32_t ndim, int32_t * restrict dim,
                            const int32_t * restrict dim_max) {
  do {
    ndim = ndim - 1;
    dim[ndim] += 1;
    if (dim[ndim] < dim_max[ndim]) {
      return true;
    } else { // reach dimension max
      if (ndim == 0) { // all dimension done
        return false;
      }
      dim[ndim] = 0;
    }
  } while(true);
}

static inline int64_t dim_to_offset(int32_t ndim, const int32_t * restrict dim,
                                    const int32_t * restrict dim_max) {
  int64_t offset = 0;
  int64_t step = 1;
  for (int32_t i = ndim - 1; i >= 0; --i) {
    offset += dim[i] * step;
    step *= dim_max[i];
  }
  return offset;
}

// If it is outside the bounds of the input, use 0.
static inline float get_value_or_zero(int32_t ndim, const int32_t * restrict dim_max,
                                      const float * restrict value, const int32_t * restrict dim, int32_t * isPad) {
  for (int32_t i = 0; i < ndim; ++i) {
    if (dim[i] < 0 || dim[i] >= dim_max[i]) {
      if(isPad){
        *isPad = 1;
      }
      return 0.f;
    }
  }
  if(isPad){
    *isPad = 0;
  }
  return value[dim_to_offset(ndim, dim, dim_max)];
}

static inline void dump_dim(int32_t ndim, int32_t * restrict dim) {
  fprintf(stderr, "  ndim: %"PRId32", X_dim: [", ndim);
  for (int i = 0; i < ndim; ++i) {
    fprintf(stderr, " %"PRId32, dim[i]);
  }
  fprintf(stderr, "]\n");
}

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
                                const int32_t * restrict strides) {
  // TODO: auto_pad

  fprintf(stderr, "Conv 2D\n");
  fprintf(stderr, "  X: %p, W: %p\n", X, W);
  fprintf(stderr, "  N: %"PRId32" C: %"PRId32" iH: %"PRId32" iW: %"PRId32"\n", N, C, iH, iW);
  fprintf(stderr, "  M: %"PRId32" kC: %"PRId32" kH: %"PRId32" kW: %"PRId32"\n", M, kC, kH, kW);
  fprintf(stderr, "  B: %p, Y: %p\n", B, Y);
  fprintf(stderr, "  oN: %"PRId32" oC: %"PRId32" oH: %"PRId32" oW: %"PRId32"\n", oN, oC, oH, oW);
  fprintf(stderr, "  auto_pad: %"PRId32"\n", auto_pad);
  fprintf(stderr, "  dilations: %p [", dilations);
  for (int i = 0; i < 2; ++i) {
    fprintf(stderr, " %"PRId32, dilations[i]);
  }
  fprintf(stderr, "]\n");
  fprintf(stderr, "  group: %"PRId32"\n", group);
  fprintf(stderr, "  kernel_shape: %p [", kernel_shape);
  if (kernel_shape != NULL) {
    for (int i = 0; i < 2; ++i) {
      fprintf(stderr, " %"PRId32, kernel_shape[i]);
    }
  }
  fprintf(stderr, "]\n");
  fprintf(stderr, "  pads: %p [", pads);
  for (int i = 0; i < 4; ++i) {
    fprintf(stderr, " %"PRId32, pads[i]);
  }
  fprintf(stderr, "]\n");
  fprintf(stderr, "  strides: %p [", strides);
  for (int i = 0; i < 2; ++i) {
    fprintf(stderr, " %"PRId32, strides[i]);
  }
  fprintf(stderr, "]\n");

  // TODO: type
  for (int32_t n = 0; n < oN; ++n) {
    for (int32_t c = 0; c < oC; ++c) {

      for (int32_t h = 0; h < oH; ++h) {
        for (int32_t w = 0; w < oW; ++w) {

          int32_t base_c = (c * group / M) * kC; // input channel <-group-> output channel
          int32_t base_h = h * strides[0] - pads[0];
          int32_t base_w = w * strides[1] - pads[1];

          float sum = 0.f;

          for (int32_t w_channel = 0; w_channel < kC; ++w_channel) {
            int32_t input_channel = base_c + w_channel;
            for (int32_t i = (base_h < 0 ? (-base_h) / dilations[0] : 0); i < kH; ++i) {
              int32_t input_h = base_h + i * dilations[0];
              if (input_h >= iH) { break; }
              for (int32_t j =  (base_w < 0 ? (-base_w) / dilations[1] : 0); j < kW; ++j) {
                int32_t input_w = base_w + j * dilations[1];
                if (input_w >= iW) { break; }

                float input = X[n][input_channel][input_h][input_w];
                float weight = W[c][w_channel][i][j];
                sum += input * weight;
              }
            }
          }

          if (B != NULL) {
            sum += B[c];
          }
          Y[n][c][h][w] = sum;
        }
      }
    }
  }
}


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
                             const int32_t * restrict strides) {
  // TODO: auto_pad

  fprintf(stderr, "Conv\n");
  fprintf(stderr, "  X: %p, W: %p\n", X, W);
  fprintf(stderr, "  ndim: %"PRId32", X_dim: %p [", ndim, X_dim);
  for (int i = 0; i < ndim; ++i) {
    fprintf(stderr, " %"PRId32, X_dim[i]);
  }
  fprintf(stderr, "]\n");
  fprintf(stderr, "  W_dim: %p [", W_dim);
  for (int i = 0; i < ndim; ++i) {
    fprintf(stderr, " %"PRId32, W_dim[i]);
  }
  fprintf(stderr, "]\n");
  fprintf(stderr, "  B: %p, Y: %p\n", B, Y);
  fprintf(stderr, "  Y_dim: %p [", Y_dim);
  for (int i = 0; i < ndim; ++i) {
    fprintf(stderr, " %"PRId32, Y_dim[i]);
  }
  fprintf(stderr, "]\n");
  fprintf(stderr, "  auto_pad: %"PRId32"\n", auto_pad);
  fprintf(stderr, "  dilations: %p [", dilations);
  for (int i = 0; i < ndim - 2; ++i) {
    fprintf(stderr, " %"PRId32, dilations[i]);
  }
  fprintf(stderr, "]\n");
  fprintf(stderr, "  group: %"PRId32"\n", group);
  fprintf(stderr, "  kernel_shape: %p [", kernel_shape);
  if (kernel_shape != NULL) {
    for (int i = 0; i < ndim - 2; ++i) {
      fprintf(stderr, " %"PRId32, kernel_shape[i]);
    }
  }
  fprintf(stderr, "]\n");
  fprintf(stderr, "  pads: %p [", pads);
  for (int i = 0; i < (ndim - 2) * 2; ++i) {
    fprintf(stderr, " %"PRId32, pads[i]);
  }
  fprintf(stderr, "]\n");
  fprintf(stderr, "  strides: %p [", strides);
  for (int i = 0; i < ndim - 2; ++i) {
    fprintf(stderr, " %"PRId32, strides[i]);
  }
  fprintf(stderr, "]\n");
  int32_t M = W_dim[0];
  int32_t C = W_dim[1];

  // TODO: type
  int32_t o_dim[ndim];
  memset(o_dim, 0, sizeof(o_dim));
  do { // while o_dim
    int32_t base_dim[ndim];
    base_dim[0] = o_dim[0]; // N
    for (int32_t i = 2; i < ndim; ++i) {
      base_dim[i] = o_dim[i] * strides[i - 2] - pads[i - 2];
    }

    float sum = 0.f;

    int32_t w_dim[ndim];
    memset(w_dim, 0, sizeof(w_dim));
    w_dim[0] = o_dim[1]; // M;
    do { // while w_dim
      if (w_dim[1] == 1) { // all D1 ~ Dn done.
        break;
      }

      int32_t i_dim[ndim];
      i_dim[0] = base_dim[0]; // N
      for (int32_t i = 2; i < ndim; ++i) {
        i_dim[i] = base_dim[i] + w_dim[i] * dilations[i - 2];
      }
      for (int32_t channel = 0; channel < C; ++channel) {
        i_dim[1] = (o_dim[1] * group / M) * C + channel; // input channel <-group-> output channel
        w_dim[1] = channel; // C

        float input = get_value_or_zero(ndim, X_dim, X, i_dim, NULL);
        float weight = get_value_or_zero(ndim, W_dim, W, w_dim, NULL);
        sum += input * weight;
      }
      w_dim[1] = 0; // reset C
    } while (next_dim(ndim, w_dim, W_dim));

    if (B != NULL) {
      sum += B[o_dim[1]];
    }
    Y[dim_to_offset(ndim, o_dim, Y_dim)] = sum;
  } while (next_dim(ndim, o_dim, Y_dim));
}

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
                             int32_t transB) {
  fprintf(stderr, "Gemm\n");
  fprintf(stderr, "  A: %p, B: %p\n", A, B);
  fprintf(stderr, "  M: %"PRId32" K: %"PRId32" N: %"PRId32"\n", M, K, N);
  fprintf(stderr, "  C: %p\n", C);
  fprintf(stderr, "  ncdim: %"PRId32", C_dim: %p\n", ncdim, C_dim);
  fprintf(stderr, "  Y: %p\n", Y);
  fprintf(stderr, "  nydim: %"PRId32", Y_dim: %p\n", nydim, Y_dim);
  fprintf(stderr, "  alpha: %f\n", alpha);
  fprintf(stderr, "  beta: %f\n", beta);
  fprintf(stderr, "  broadcast: %"PRId32"\n", broadcast);
  fprintf(stderr, "  transA: %"PRId32"\n", transA);
  fprintf(stderr, "  transB: %"PRId32"\n", transB);
  // TODO: broadcast
  // A: M x K
  // B: K x N
  // C: M x N
  for (int32_t i = 0; i < M; ++i) {
    for (int32_t j = 0; j < N; ++j) {
      float sum = 0.f;
      for (int32_t k = 0; k < K; ++k) {
        sum += A[(transA ? (k * M + i) : (i * K + k))]
            * B[(transB ? (j * K + k) : (k * N + j))];
      }
      Y[i * N + j] = sum * alpha + C[i * N + j] * beta;
    }
  }
}

void ONNC_RUNTIME_maxpool_float(void * restrict onnc_runtime_context,
                                const float * restrict X,
                                int32_t ndim, const int32_t * restrict X_dim,
                                float * restrict Y,
                                const int32_t * restrict Y_dim,
                                int32_t auto_pad,
                                const int32_t * restrict kernel_shape,
                                const int32_t * restrict pads,
                                const int32_t * restrict strides) {
  // TODO auto_pad
  fprintf(stderr, "MaxPool\n");
  fprintf(stderr, "  X: %p\n", X);
  fprintf(stderr, "  ndim: %"PRId32", X_dim: %p [", ndim, X_dim);
  for (int i = 0; i < ndim; ++i) {
    fprintf(stderr, " %"PRId32, X_dim[i]);
  }
  fprintf(stderr, "]\n");
  fprintf(stderr, "  Y: %p\n", Y);
  fprintf(stderr, "  Y_dim: %p [", Y_dim);
  for (int i = 0; i < ndim; ++i) {
    fprintf(stderr, " %"PRId32, Y_dim[i]);
  }
  fprintf(stderr, "]\n");
  fprintf(stderr, "  auto_pad: %"PRId32"\n", auto_pad);
  fprintf(stderr, "  kernel_shape: %p [", kernel_shape);
  for (int i = 0; i < ndim - 2; ++i) {
    fprintf(stderr, " %"PRId32, kernel_shape[i]);
  }
  fprintf(stderr, "]\n");
  fprintf(stderr, "  pads: %p [", pads);
  for (int i = 0; i < (ndim - 2) * 2; ++i) {
    fprintf(stderr, " %"PRId32, pads[i]);
  }
  fprintf(stderr, "]\n");
  fprintf(stderr, "  strides: %p [", strides);
  for (int i = 0; i < ndim - 2; ++i) {
    fprintf(stderr, " %"PRId32, strides[i]);
  }
  fprintf(stderr, "]\n");
  int32_t o_dim[ndim];
  memset(o_dim, 0, sizeof(o_dim));
  do { // while o_dim
    int32_t base_dim[ndim];
    for (int32_t i = 2; i < ndim; ++i) {
      base_dim[i] = o_dim[i] * strides[i - 2] - pads[i - 2];
    }

    float max = -FLT_MAX;

    int32_t k_dim[ndim - 2];
    memset(k_dim, 0, sizeof(k_dim));
    do { // while k_dim
      int32_t i_dim[ndim];
      i_dim[0] = o_dim[0]; // N
      i_dim[1] = o_dim[1]; // C
      for (int32_t i = 2; i < ndim; ++i) {
        i_dim[i] = base_dim[i] + k_dim[i - 2];
      }
      float input = get_value_or_zero(ndim, X_dim, X, i_dim, NULL);
      max = fmaxf(input, max);
    } while (next_dim(ndim - 2, k_dim, kernel_shape));

    Y[dim_to_offset(ndim, o_dim, Y_dim)] = max;
  } while (next_dim(ndim, o_dim, Y_dim));
}

void ONNC_RUNTIME_relu_float(void * restrict onnc_runtime_context,
                             const float * restrict X,
                             int32_t ndim, const int32_t * restrict X_dim,
                             float * restrict Y) {
  fprintf(stderr, "Relu\n");
  fprintf(stderr, "  X: %p\n", X);
  fprintf(stderr, "  ndim: %"PRId32", X_dim: %p\n", ndim, X_dim);
  fprintf(stderr, "  Y: %p\n", Y);
  int64_t size = 1;
  for (int32_t i = 0; i < ndim; ++i) {
    size *= X_dim[i];
  }
  fprintf(stderr, "  X[0]: %f\n", X[0]);
  fprintf(stderr, "  X[1]: %f\n", X[1]);
  fprintf(stderr, "  X[2]: %f\n", X[2]);
  for (int64_t i = 0; i < size; ++i) {
    Y[i] = X[i] < 0.f ? 0.f : X[i];
  }
  fprintf(stderr, "  Y[0]: %f\n", Y[0]);
  fprintf(stderr, "  Y[1]: %f\n", Y[1]);
  fprintf(stderr, "  Y[2]: %f\n", Y[2]);
}

void ONNC_RUNTIME_softmax_float(void * restrict onnc_runtime_context,
                                const float * restrict input,
                                int32_t ndim, const int32_t * restrict input_dim,
                                int32_t axis,
                                float * restrict output) {
  fprintf(stderr, "Softmax\n");
  fprintf(stderr, "  input %p\n", input);
  fprintf(stderr, "  ndim: %"PRId32", input_dim: %p [", ndim, input_dim);
  for (int i = 0; i < ndim; ++i) {
    fprintf(stderr, " %"PRId32, input_dim[i]);
  }
  fprintf(stderr, "]\n");
  fprintf(stderr, "  axis: %"PRId32"\n", axis);
  fprintf(stderr, "  output %p\n", output);
  int64_t N = 1;
  for (int32_t i = 0; i < axis; ++i) {
    N *= input_dim[i];
  }
  int64_t D = 1;
  for (int32_t i = axis; i < ndim; ++i) {
    D *= input_dim[i];
  }
  fprintf(stderr, "  N: %"PRId64" D: %"PRId64"\n", N, D);
  fprintf(stderr, "  input[0]: %f\n", input[0]);
  fprintf(stderr, "  input[1]: %f\n", input[1]);
  fprintf(stderr, "  input[2]: %f\n", input[2]);
  for (int64_t i = 0; i < N; ++i) { // Y = exp(X - max(X)) / sum(exp(X - max(X)))
    const float *X = input + i * D;
    float *Y = output + i * D;

    float max = -FLT_MAX;
    for (int64_t j = 0; j < D; ++j) {
      max = fmaxf(X[j], max);
    }

    float sum = 0.f;
    for (int64_t j = 0; j < D; ++j) {
      float v = expf(X[j] - max);
      sum += v;
      Y[j] = v;
    }
    fprintf(stderr, "  max: %f\n", max);
    fprintf(stderr, "  sum: %f\n", sum);

    for (int64_t j = 0; j < D; ++j) {
      Y[j] /= sum;
    }
  }
  fprintf(stderr, "  output[0]: %f\n", output[0]);
  fprintf(stderr, "  output[1]: %f\n", output[1]);
  fprintf(stderr, "  output[2]: %f\n", output[2]);
}

void ONNC_RUNTIME_reshape_float(void * restrict onnc_runtime_context,
                                const float * restrict data,
                                int32_t ndim, const int32_t * restrict X_dim,
                                float * restrict reshaped) {
  int64_t size = 1;
  for (int32_t i = 0; i < ndim; ++i) {
    size *= X_dim[i];
  }
  memcpy(reshaped, data, size * sizeof(float));
}


void ONNC_RUNTIME_lrn_float(void * restrict onnc_runtime_context,
                            const float * restrict X,
                            int32_t ndim, const int32_t * restrict X_dim,
                            float alpha,
                            float beta,
                            float bias,
                            int32_t size,
                            float * restrict Y) {
  fprintf(stderr, "LRN\n");
  fprintf(stderr, "  X: %p\n", X);
  fprintf(stderr, "  ndim: %"PRId32", X_dim: %p\n", ndim, X_dim);
  fprintf(stderr, "  alpha: %f\n", alpha);
  fprintf(stderr, "  beta: %f\n", beta);
  fprintf(stderr, "  bias: %f\n", bias);
  fprintf(stderr, "  size: %"PRId32"\n", size);
  fprintf(stderr, "  Y: %p\n", Y);
  // XXX: WFT ONNX.
  // (bias+(alpha/size)*sum(xi^2 for every xi in the local region))^beta
  float alpha_over_size = alpha / size;

  int64_t N = X_dim[0];
  int64_t C = X_dim[1];
  int64_t len = 1;
  for (int32_t i = 2; i < ndim; ++i) {
    len *= X_dim[i];
  }
  for (int64_t n = 0; n < N; ++n) {
    for (int64_t c = 0; c < C; ++c) {
      for (int64_t i = 0; i < len; ++i) {
        int64_t start = c - (size/2);
        if (start < 0) { start = 0; }
        int64_t end = c + (size/2);
        if (end >= C) { end = C - 1; }

        float sum = 0.f;
        for (int64_t j = start; j <= end; ++j) {
          float value = X[(n*C + j)*len + i];
          sum += value * value;
        }
        Y[(n*C + c)*len + i] = X[(n*C + c)*len + i] * powf(bias + alpha_over_size * sum, -beta);
      }
    }
  }
}

void ONNC_RUNTIME_add_float(void * restrict onnc_runtime_context,
                            const float * restrict A,
                            int32_t ndim, const int32_t * restrict A_dim,
                            const float * restrict B,
                            float * restrict C) {
  int64_t size = 1;
  for (int32_t i = 0; i < ndim; ++i) {
    size *= A_dim[i];
  }
  for (int64_t i = 0; i < size; ++i) {
    C[i] = A[i] + B[i];
  }
}

void ONNC_RUNTIME_averagepool_float(void * restrict onnc_runtime_context,
                                    const float * restrict X,
                                    int32_t ndim, const int32_t * restrict X_dim,
                                    float * restrict Y,
                                    const int32_t * restrict Y_dim,
                                    int32_t auto_pad,
                                    int32_t count_include_pad,
                                    const int32_t * restrict kernel_shape,
                                    const int32_t * restrict pads,
                                    const int32_t * restrict strides) {
  // TODO auto_pad
  fprintf(stderr, "AveragePool\n");
  fprintf(stderr, "  X: %p\n", X);
  fprintf(stderr, "  ndim: %"PRId32", X_dim: %p [", ndim, X_dim);
  for (int i = 0; i < ndim; ++i) {
    fprintf(stderr, " %"PRId32, X_dim[i]);
  }
  fprintf(stderr, "]\n");
  fprintf(stderr, "  Y: %p\n", Y);
  fprintf(stderr, "  Y_dim: %p [", Y_dim);
  for (int i = 0; i < ndim; ++i) {
    fprintf(stderr, " %"PRId32, Y_dim[i]);
  }
  fprintf(stderr, "]\n");
  fprintf(stderr, "  auto_pad: %"PRId32"\n", auto_pad);
  fprintf(stderr, "  count_include_pad: %"PRId32"\n", count_include_pad);
  fprintf(stderr, "  kernel_shape: %p [", kernel_shape);
  for (int i = 0; i < ndim - 2; ++i) {
    fprintf(stderr, " %"PRId32, kernel_shape[i]);
  }
  fprintf(stderr, "]\n");
  fprintf(stderr, "  pads: %p [", pads);
  for (int i = 0; i < (ndim - 2) * 2; ++i) {
    fprintf(stderr, " %"PRId32, pads[i]);
  }
  fprintf(stderr, "]\n");
  fprintf(stderr, "  strides: %p [", strides);
  for (int i = 0; i < ndim - 2; ++i) {
    fprintf(stderr, " %"PRId32, strides[i]);
  }
  fprintf(stderr, "]\n");

  int64_t size = 1;
  for (int i = 0; i < ndim - 2; ++i) {
    size *= kernel_shape[i];
  }

  int32_t o_dim[ndim];
  memset(o_dim, 0, sizeof(o_dim));
  do { // while o_dim
    int32_t base_dim[ndim];
    for (int32_t i = 2; i < ndim; ++i) {
      base_dim[i] = o_dim[i] * strides[i - 2] - pads[i - 2];
    }

    float sum = 0.f;

    int32_t k_dim[ndim - 2];
    memset(k_dim, 0, sizeof(k_dim));
    int32_t padCount = 0;
    do { // while k_dim
      int32_t i_dim[ndim];
      i_dim[0] = o_dim[0]; // N
      i_dim[1] = o_dim[1]; // C
      for (int32_t i = 2; i < ndim; ++i) {
        i_dim[i] = base_dim[i] + k_dim[i - 2];
      }
      int32_t isPad = 0;
      sum += get_value_or_zero(ndim, X_dim, X, i_dim, &isPad);
      if(isPad){
        ++padCount;
      }
    } while (next_dim(ndim - 2, k_dim, kernel_shape));
    if (count_include_pad) {
      sum /= size;
    } else {
      sum /= (size - padCount);
    }

    Y[dim_to_offset(ndim, o_dim, Y_dim)] = sum;
  } while (next_dim(ndim, o_dim, Y_dim));
}

void ONNC_RUNTIME_batchnormalization_float(void * restrict onnc_runtime_context,
                                    const float * restrict X,
                                    int32_t ndim, const int32_t * restrict X_dim,
                                    float * restrict Y,
                                    const float * restrict scale,
                                    const float * restrict B,
                                    const float * restrict meanI,
                                    const float * restrict varI,
                                    float * restrict meanO,
                                    float * restrict varO,
                                    float * restrict saved_mean,
                                    float * restrict saved_var,
                                    float epsilon,
                                    float momentum,
                                    int32_t spatial) {
  // Preparation
  int32_t xN = X_dim[0], xC = X_dim[1];
  fprintf(stderr, "Batchnormalization\n");
  fprintf(stderr, "  X: %p\n", X);
  fprintf(stderr, "  epsilon: %f\n", epsilon);
  fprintf(stderr, "  momentum: %f\n", momentum);
  fprintf(stderr, "  spatial: %"PRId32"\n", spatial);
  fprintf(stderr, "  ndim: %"PRId32", X_dim: %p [", ndim, X_dim);
  for (int i = 0; i < ndim; ++i) {
    fprintf(stderr, " %"PRId32, X_dim[i]);
  }
  fprintf(stderr, "]\n");
  fprintf(stderr, "  Y: %p\n", Y);
  fprintf(stderr, "  scale: %p [", scale);
  for (int i = 0; i < 2; ++i) {
    fprintf(stderr, " %f", scale[i]);
  }
  fprintf(stderr, "]\n");
  fprintf(stderr, "  B: %p [", B);
  for (int i = 0; i < 2; ++i) {
    fprintf(stderr, " %f", B[i]);
  }
  fprintf(stderr, "]\n");
  fprintf(stderr, "  meanI: %p [", meanI);
  for (int i = 0; i < 2; ++i) {
    fprintf(stderr, " %f", meanI[i]);
  }
  fprintf(stderr, "]\n");
  fprintf(stderr, "  varI: %p [", varI);
  for (int i = 0; i < 2; ++i) {
    fprintf(stderr, " %f", varI[i]);
  }
  fprintf(stderr, "]\n");
  // TODO: spatial
  int32_t strideSize = 1;
  for(int32_t i = 2; i < ndim; ++i){
    strideSize *= X_dim[i];
  }

  for(int32_t iN = 0; iN < xN; ++iN){
    for(int32_t iC = 0; iC < xC; ++iC){
      const float *pIMean = meanI + iN * xC;
      const float *pIVariance = varI + iN * xC;
      const float *pX = X + iN * xC * strideSize + iC * strideSize;
      float *pY = Y + iN * xC * strideSize + iC * strideSize;
      // Output
      for(int32_t i = 0; i < strideSize; ++i){
        pY[i] = scale[iC] * (pX[i] - pIMean[iC]) / sqrtf(pIVariance[iC] + epsilon) + B[iC];
      }
    }
  }
}