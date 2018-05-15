#define restrict __restrict__
extern "C" {
#include "onnc-runtime-internal.h"
}
#undef restrict

#include <stdlib.h>
#include <stdio.h>

int main(int argc, const char *argv[]) {
  if (argc < 2) {
    printf("%s ONNC_WEIGHT\n", argv[0]);
    return EXIT_SUCCESS;
  }
  void *context = ONNC_RUNTIME_init_runtime(argv[1]);
  
  float *X = new float[35]{
    0., 1., 2., 3., 4.,
    5., 6., 7., 8., 9.,
    10., 11., 12., 13., 14.,
    15., 16., 17., 18., 19.,
    20., 21., 22., 23., 24.,
    25., 26., 27., 28., 29.,
    30., 31., 32., 33., 34
  };
  float *W = new float[9]{
    1., 1., 1.,
    1., 1., 1.,
    1., 1., 1.
  };
  float *Y = new float[25];
  ONNC_RUNTIME_conv_2d_float(context,
                             1, 1, 7, 5,
                             X,
                             1, 1, 3, 3,
                             W,
                             NULL,
                             1, 1, 4, 2,
                             Y,
                             0,
                             new int32_t[2]{1, 1},
                             1,
                             new int32_t[2]{3, 3},
                             new int32_t[4]{1, 0, 1, 0},
                             new int32_t[2]{2, 2});
  for (int i = 0; i < 25; ++i) {
    printf("%f\n", Y[i]);
  }
  if (ONNC_RUNTIME_shutdown_runtime(context)) {
    return EXIT_SUCCESS;
  } else {
    return EXIT_FAILURE;
  }
}
