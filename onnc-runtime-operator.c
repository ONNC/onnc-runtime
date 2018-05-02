#include "onnc-runtime-internal.h"

#include <assert.h>
#include <string.h>

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
    step += dim_max[i];
  }
  return offset;
}

// If it is outside the bounds of the input, use 0.
static inline float get_value_or_zero(int32_t ndim, const int32_t * restrict dim_max,
                                      const float * restrict value, const int32_t * restrict dim) {
  for (int32_t i = 0; i < ndim; ++i) {
    if (dim[i] < 0 || dim[i] >= dim_max[i]) {
      return 0.f;
    }
  }
  return value[dim_to_offset(ndim, dim, dim_max)];
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
  // TODO: auto_pad, group, B (bias)

  assert(X_dim[1] == W_dim[1]); // C
  int32_t C = X_dim[1];

  // TODO: type
  int32_t o_dim[ndim];
  memset(o_dim, 0, sizeof(o_dim));
  while (next_dim(ndim, o_dim, Y_dim)) {
    int32_t center_dim[ndim];
    center_dim[0] = o_dim[0]; // N
    for (uint32_t i = 2; i < ndim; ++i) {
      center_dim[i] = o_dim[i] * strides[i - 2] - pads[i - 2];
    }

    float sum = 0.f;

    int32_t w_dim[ndim];
    memset(w_dim, 0, sizeof(w_dim));
    w_dim[0] = o_dim[1]; // M;
    while (next_dim(ndim, w_dim, W_dim)) {
      if (w_dim[1] == 1) { // all D1 ~ Dn done.
        break;
      }

      int32_t i_dim[ndim];
      i_dim[0] = center_dim[0]; // N
      for (uint32_t i = 2; i < ndim; ++i) {
        i_dim[i] = center_dim[i] + w_dim[i] * dilations[i - 2];
      }
      for (int32_t channel = 0; channel < C; ++channel) {
        i_dim[1] = channel; // C
        w_dim[1] = channel; // C

        float input = get_value_or_zero(ndim, X_dim, X, i_dim);
        float weight = get_value_or_zero(ndim, W_dim, W, w_dim);
        sum += input * weight;
      }
      w_dim[1] = 0; // reset C
    } // while w_dim

    Y[dim_to_offset(ndim, o_dim, Y_dim)] = sum;
  } // while o_dim
}

