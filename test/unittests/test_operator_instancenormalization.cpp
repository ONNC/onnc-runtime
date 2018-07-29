#include <skypat/skypat.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define restrict __restrict__
extern "C"{
#include <operator/instancenormalization.h>
}
#undef restrict

SKYPAT_F(Operator_InstanceNormalization, non_broadcast){
    // Prepare
    srand(time(NULL));
    
    int32_t ndim = rand() % 3 + 1;
    int32_t dims[ndim];
    int32_t dims2[1];

    int32_t dataSize =1;
    for(int32_t i = 0; i < ndim; ++i){
        dims[i] = rand() % 20;
        dataSize *= dims[i];
    }
    dims2[0] = dims[1];

    float epsilon = 1e-5f;

    float A[dataSize];
    float B[dims[1]];
    float C[dims[1]];
    float D[dataSize];

    for(int32_t i = 0; i < dataSize; ++i){
        //x
        A[i] = rand() / (float)RAND_MAX;
    }

    for(int32_t i = 0; i < dims[1]; ++i){
        //scale
        B[i] = rand() / (float)RAND_MAX + 1.0f;
        //bias
        C[i] =rand() / (float)RAND_MAX;
    }

    // Run
    ONNC_RUNTIME_instancenormalization_float(
        NULL
        ,A
        ,ndim,dims
        ,B
        ,1,dims2
        ,C
        ,1,dims2
        ,D
        ,ndim,dims
        ,epsilon
    );
    // Check
    //Not Practical
    printf("IN :\n");
    for(int32_t i = 0; i < dataSize; ++i){
        printf("%f \n", D[i]);
    }
    printf("size %d \n", dataSize);
}

int main(int argc, char *argv[]){
    skypat::Test::Initialize(argc, argv);
    skypat::Test::RunAll();
    return (skypat::testing::UnitTest::self()->getNumOfFails() == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
