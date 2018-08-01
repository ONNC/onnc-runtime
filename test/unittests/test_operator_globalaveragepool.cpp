#include <skypat/skypat.h>
#include <cstdlib>
#include <ctime>
#include <cmath>
#include <stdio.h>

#define restrict __restrict__
extern "C"{
#include <operator/globalaveragepool.h>
}
#undef restrict

SKYPAT_F(Operator_GlobalAveragePool, non_broadcast){
    // Prepare
    float A[96];
    int32_t ndim = 4;
    int32_t dims[4] = {2,3,4,4};
    int32_t ndim2 = 2;
    int32_t dims2[2] = {2,3};
    for(int32_t i = 0; i < 96; i++)
    {
        A[i] = i;
    }
    float B[6];
    // Run
    ONNC_RUNTIME_globalaveragepool_float(NULL
        ,A
        ,ndim,dims
        ,B
        ,ndim2,dims2
    );
    // Check
    /*
    [[ 7.5 23.5 39.5]
    [55.5 71.5 87.5]]

    code:
        import numpy as np
        arr = np.arange(96).reshape(2,3,4,4)
        axis = (2,3) //LEFT N x C 
        newArr = np.average(arr,axis=axis)
        print(newArr)
    */
    printf("Sample :\n");
    for(int32_t i = 0; i < 6; i++){
        printf("%10f \n", B[i]);
    }
    printf("%d datas:\n", 6);
}

int main(int argc, char *argv[]){
    skypat::Test::Initialize(argc, argv);
    skypat::Test::RunAll();
    return (skypat::testing::UnitTest::self()->getNumOfFails() == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}