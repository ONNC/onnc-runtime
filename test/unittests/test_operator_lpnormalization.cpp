#include <skypat/skypat.h>
#include <cstdlib>
#include <cmath>
#include <stdio.h>

#define restrict __restrict__
extern "C"{
#include <operator/lpnormalization.h>
}
#undef restrict

SKYPAT_F(Operator_Lpnormalization, non_broadcast){
    //prepare
    float A[36];

    for(int32_t i = 0; i < 36; i++){
        A[i] = i;
    }
    int32_t ndim = 4;
    int32_t dims[4] = {1,2,3,6};
    int32_t ndim2 = 3;
    int32_t dims2[3] = {1,2,6};
    float B[12];

    int32_t p = 2;
    int32_t axis = 2;

    int32_t outputSize = 12;
    ONNC_RUNTIME_lpnormalization_float(
        NULL
        ,A
        ,ndim,dims
        ,B
        ,ndim2,dims2
        ,axis
        ,p
    );
    // Check
    // Testing is not practical
    //For reference here is the python code to verify it
    /*
        import numpy as np
        from numpy import linalg as LA
        arr = np.arange(36).reshape(1,2,3,6)
        newArr = LA.norm(arr, 2, axis=2)
        newArr = np.reshape(newArr , 12)
        print (newArr)
    */
    printf("Sample :\n")
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