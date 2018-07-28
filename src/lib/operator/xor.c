#include <operator/xor.h>

#include <stdint.h>
#include <stdbool.h>

//function declaration
static void tile_float(
  const float * restrict input_input
  ,int32_t input_input_ndim, const int32_t * restrict input_input_dims
  ,const float * restrict input_repeats
  ,float * restrict output_output
  ,int32_t output_output_ndim, const int32_t * restrict output_output_dims
);

void ONNC_RUNTIME_xor_float(
  void * restrict onnc_runtime_context
  ,const float * restrict input_A
  ,int32_t input_A_ndim, const int32_t * restrict input_A_dims
  ,const float * restrict input_B
  ,int32_t input_B_ndim, const int32_t * restrict input_B_dims
  ,float * restrict output_C
  ,int32_t output_C_ndim, const int32_t * restrict output_C_dims
  
) {
  int32_t new_ndim = (input_A_ndim >= input_B_ndim) ? input_A_ndim : input_B_ndim;
  int32_t new_dims_A[new_ndim];
  int32_t new_dims_B[new_ndim];
  int32_t dims_C[new_ndim];

  int32_t offsetA = new_ndim - input_A_ndim;
  int32_t offsetB = new_ndim - input_B_ndim;

  //for new dimensions A
  for(int32_t i = 0; i < offsetA; i++){
    //extend dimension
    new_dims_A[i] = 1;
  }
  for(int32_t i = offsetA; i < new_ndim; i++){
    //shift dimension
    new_dims_A[i] = input_A_dims[i - offsetA];
  }
  //for new dimensions B
  for(int32_t i = 0; i < offsetB; i++){
    //extend dimension
    new_dims_B[i] = 1;
  }
  for(int32_t i = offsetB; i < new_ndim; i++){
    //shift dimension
    new_dims_B[i] = input_B_dims[i - offsetB];
  }

  //Get the tiling from A's perspective
  float permA[new_ndim];
  for(int32_t i = 0; i < new_ndim; i++){
    permA[i] = 1;
    if(new_dims_A[i] == 1 && new_dims_B[i] != 1){
      permA[i] = new_dims_B[i];
    }
  }
  
  //Get the tiling from B's perspective
  float permB[new_ndim];
  for(int32_t i = 0; i < new_ndim; i++){
    permB[i] = 1;
    if(new_dims_B[i] == 1 && new_dims_A[i] != 1){
      permB[i] = new_dims_A[i];
    }
  }

  //get dataSize
  int32_t elem = 1;
  int32_t tiledDims[new_ndim];
  for(int32_t i = 0; i < new_ndim; i++){
    elem *= ((new_dims_A[i] >= new_dims_B[i]) ? new_dims_A[i] : new_dims_B[i]);
    tiledDims[i] = (new_dims_A[i] >= new_dims_B[i]) ? new_dims_A[i] : new_dims_B[i];
  }

  //tile A
  float tiledA[elem];
  tile_float(
    (const float*)input_A,
    (int32_t)new_ndim, (const int32_t*)new_dims_A,
    (const float *)permA,
    (float *)tiledA,
    (int32_t)new_ndim, (const int32_t*)tiledDims
  );

  //tile B
  float tiledB[elem];
  tile_float(
    (const float*)input_B,
    (int32_t)new_ndim, (const int32_t*)new_dims_B,
    (const float *)permB,
    (float *)tiledB,
    (int32_t)new_ndim, (const int32_t*)tiledDims
  );

  //after tiling both array , they can do broadcast operations,this one is xor
  for(int32_t i = 0; i < elem; i++){
    if((bool)tiledA[i] != (bool)tiledB[i]){
      output_C[i] = (float)true;
    } else{
      output_C[i] = (float)false;
    }
  }
}

static void tile_float(
  const float * restrict input_input
  ,int32_t input_input_ndim, const int32_t * restrict input_input_dims
  ,const float * restrict input_repeats
  ,float * restrict output_output
  ,int32_t output_output_ndim, const int32_t * restrict output_output_dims
){
  //calculate new dimension
  int32_t ndim = input_input_ndim;
  int32_t newDims[ndim];
  for (int32_t i = 0; i < ndim; i++){
    //By definition input_repeats should be an integer
    newDims[i] = input_input_dims[i] * (int32_t)input_repeats[i];
  }

  //output elements
  int32_t elem = 1;
  for(int32_t i = 0; i < ndim; i++){
    elem *= newDims[i];
  }

  //input stride, for reference
  int32_t dimStride[ndim];
  dimStride[ndim-1] = 1;
  for(int32_t i = (ndim - 1) - 1; i >= 0; i--){
    dimStride[i] = dimStride[i+1] * input_input_dims[i+1];
  }

  

  //Tiling process starts here

  //output stride, for calculating new coordinates
  int32_t newDimStride[ndim];
  newDimStride[ndim-1] = 1;
  for(int32_t i = (ndim - 1) - 1; i >= 0; i--){
    newDimStride[i] = newDimStride[i+1] * newDims[i+1];
  }

  //To locate coordinates, count the strides!
  int32_t coor[elem][ndim];
  int32_t dimCounter = 0;
  int32_t indexCounter =0;
  for(int32_t i = 0; i < elem; i++){
      indexCounter = i;
      dimCounter = 0;
      while(dimCounter < ndim){
          coor[i][dimCounter] = indexCounter / newDimStride[dimCounter];
          indexCounter %= newDimStride[dimCounter];
          dimCounter++;
      }
  }

  //every new coordinate correspond to original input offset
  int32_t correspondIndex = 0;
  for(int32_t i = 0; i < elem; i++){
    correspondIndex = 0;
    for(int32_t j = 0; j < ndim; j++){
      //mapping operation
      correspondIndex += ((coor[i][j] % input_input_dims[j]) * dimStride[j]);
    }
    //output
    output_output[i] = input_input[correspondIndex];
  }
}
