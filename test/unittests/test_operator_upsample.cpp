#include <skypat/skypat.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define restrict __restrict__
extern "C"{
#include <operator/upsample.h>
}
#undef restrict

SKYPAT_F(Operator_Upsample, non_broadcast){
    // Prepare
    srand(time(NULL));
    int32_t ndim = 2;
    int32_t dims[2] = {
        2, 2
    };
    int32_t dataSize = 16;
    float A[4] = {
        0, 1, 2, 3
    };
    float B[dataSize];
    float Ans[16] = {
        0, 0, 1, 1,
        0, 0, 1, 1,
        2, 2, 3, 3,
        2, 2, 3, 3
    };
    float scales[2] = {
        2, 2
    };
    int32_t odims[2] = {
        4, 4
    };
    // Run
    ONNC_RUNTIME_upsample_float(NULL
        ,A
        ,ndim,dims
        ,B
        ,ndim,odims,
        NULL,
        scales,
        2
    );
    // Check
    for(int32_t i = 0; i < dataSize; ++i){
        EXPECT_EQ(B[i], Ans[i]);
    }
}

int main(int argc, char *argv[]){
    skypat::Test::Initialize(argc, argv);
    skypat::Test::RunAll();
    return (skypat::testing::UnitTest::self()->getNumOfFails() == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
