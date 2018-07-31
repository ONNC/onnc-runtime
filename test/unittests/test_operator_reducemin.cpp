#include <skypat/skypat.h>
#include <cstdlib>
#include <cmath>
#include <stdio.h>

#define restrict __restrict__
extern "C"{
#include <operator/reducemin.h>
}
#undef restrict

SKYPAT_F(Operator_ReduceMin, non_broadcast){
    //prepare
    float A[48];
    for(int32_t i = 0; i < 48; i++){
        A[i] = i;
    }
    int32_t ndim = 4;
    int32_t dims[4] = {2,3,2,4};
    
    int32_t number_of_axes = 2;
    int32_t axes[2] = {0,3};

    int32_t ndim2 = 2;
    int32_t dims2[2] = {3,2};
    float B[6];
    
    int32_t keepdim = 0;

    int32_t outputSize = 6;
    ONNC_RUNTIME_reducemin_float(
        NULL
        ,A
        ,ndim,dims
        ,B
        ,ndim2,dims2
        ,axes
        ,number_of_axes
        ,keepdim
    );
    // Check
    // Testing is not practical
    //For reference here is the python code to verify it
    /*
        import numpy as np
        arr = np.arange(48).reshape(2,3,2,4)
        axistuple = (0,3)
        output = np.amax(arr, axis=axistuple, keepdims=0)
        print(output)
    */
    printf("Sample :\n");
    for(int32_t i = 0; i < outputSize; i++){
        printf("%10f \n", B[i]);
    }
    printf("%d datas:\n", outputSize);
}

int main(int argc, char *argv[]){
    skypat::Test::Initialize(argc, argv);
    skypat::Test::RunAll();
    return (skypat::testing::UnitTest::self()->getNumOfFails() == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}