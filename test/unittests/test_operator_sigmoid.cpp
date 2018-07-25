#include <skypat/skypat.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define restrict __restrict__
extern "C"{
#include <operator/sigmoid.h>
}
#undef restrict

SKYPAT_F(Operator_Sigmoid, non_broadcast){
    // Prepare
    srand(time(NULL));
    int32_t ndim = rand() % 3 + 1;
    int32_t dims[ndim];
    int32_t dataSize = 1;
    for(int32_t i = 0; i < ndim; ++i){
        dims[i] = rand() % 100 + 1;
        dataSize *= dims[i];
    }
    float A[dataSize], B[dataSize], Ans[dataSize];
    for(int32_t i = 0; i < dataSize; ++i){
        A[i] = rand() % 1000 / 100;
        Ans[i] = 1.0f / (1.0f + expf(A[i] * -1.0f));
    }
    // Run
    ONNC_RUNTIME_sigmoid_float(NULL
        ,A
        ,ndim,dims
        ,B
        ,ndim,dims
    );
    // Check
    for(int32_t i = 0; i < dataSize; ++i){
        EXPECT_TRUE(fabsf(B[i] - Ans[i]) < 1e-7f);
    }
}

int main(int argc, char *argv[]){
    skypat::Test::Initialize(argc, argv);
    skypat::Test::RunAll();
    return (skypat::testing::UnitTest::self()->getNumOfFails() == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
