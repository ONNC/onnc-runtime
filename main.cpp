#define restrict __restrict__
extern "C" {
#include "onnc-runtime-internal.h"
}
#undef restrict

#include <stdlib.h>
#include <stdio.h>

int main(int argc, const char *argv[]) {
  /*if (argc < 2) {
    printf("%s ONNC_WEIGHT\n", argv[0]);
    return EXIT_SUCCESS;
  }*/
  //void *context = ONNC_RUNTIME_init_runtime(argv[1]);
  
  float *A = new float[30]{
    0., 1.,
    2., 2.,
    4., 5.,
    
    6., 7.,
    8., 9.,
    10., 11.,
    
    12., 13.,
    14., 15.,
    16., 17.,
    
    18., 19.,
    20., 21.,
    22., 23.,
    
    24., 25.,
    26., 27.,
    28., 29.
  };
  float *B = new float[30]{
    0., 1.,
    2., 2.,
    4., 5.,
    
    6., 7.,
    8., 9.,
    10., 11.,
    
    12., 13.,
    14., 15.,
    16., 17.,
    
    18., 19.,
    20., 21.,
    22., 23.,
    
    24., 25.,
    26., 27.,
    28., 29.
  };
  int32_t x_dim[4] = {5, 1, 2, 3}; 
  float *Y = new float[30];
  ONNC_RUNTIME_add_float(NULL,
                             A, 4, x_dim,
                             B,
                             Y);
  
  // Test Output
  printf("== Y ==\n");
  for(int32_t j = 0; j < 30; ++j){
    printf("%f ", Y[j]);
  }
  //if (ONNC_RUNTIME_shutdown_runtime(context)) {
    return EXIT_SUCCESS;
  //} else {
  //  return EXIT_FAILURE;
  //}
}
