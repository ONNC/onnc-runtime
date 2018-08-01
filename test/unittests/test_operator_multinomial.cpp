#include <skypat/skypat.h>
#include <cstdlib>
#include <cmath>
#include <stdio.h>

#define restrict __restrict__
extern "C"{
#include <operator/multinomial.h>
}
#undef restrict

SKYPAT_F(Operator_Multinomial, non_broadcast){
    //prepare
    float A[6] = {0.2f, 0.3f, 0.5f, 0.15f, 0.45f, 0.4f};
    int32_t ndim = 2;
    int32_t dims[2] = {2,3};

    float B[6];

    int32_t sampleSize = 100;
    float seed = 1012123213;
    ONNC_RUNTIME_multinomial_float(
        NULL
        ,A
        ,ndim,dims
        ,B
        ,ndim,dims
        ,0
        ,sampleSize
        ,seed
    );

    int32_t outputSize = 6;
    // Check
    // Testing is not practical
    //For reference here is the python code to verify it
    printf("Sample :\n");
    for(int32_t i = 0; i < outputSize; i++){
        printf("%10f \n", B[i]);
    }
    printf("%d datas\n", outputSize);
}

int main(int argc, char *argv[]){
    skypat::Test::Initialize(argc, argv);
    skypat::Test::RunAll();
    return (skypat::testing::UnitTest::self()->getNumOfFails() == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}