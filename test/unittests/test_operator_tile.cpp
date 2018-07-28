#include <skypat/skypat.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define restrict __restrict__
extern "C"{
#include <operator/tile.h>
}
#undef restrict

SKYPAT_F(Operator_Not, non_broadcast){
    // Prepare
    int32_t ndim = 3;
    int32_t dims[3] = {2,2,2};
    float A[8] = {0,1,2,3,4,5,6,7};
    float rep[3] = {2,2,2};
    float B[64];
    int32_t tileDims[3] = {4,4,4};
    float Ans[64] = {0,1,0,1,2,3,2,3,0,1,0,1,2,3,2,3,4,5,4,5,6,7,6,7,4,5,4,5,6,7,6,7,0,1,0,1,2,3,2,3,0,1,0,1,2,3,2,3,4,5,4,5,6,7,6,7,4,5,4,5,6,7,6,7};

    int dataSize = 64;
    // Run
    ONNC_RUNTIME_tile_float(NULL,
        (const float*)A,
        (int32_t)ndim, (const int32_t*)dims,
        (const float *)rep,
        (int32_t)ndim, (const int32_t*)dims,
        (float *)B,
        (int32_t)ndim, (const int32_t*)tileDims
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