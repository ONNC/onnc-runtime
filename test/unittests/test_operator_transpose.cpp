#include <skypat/skypat.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define restrict __restrict__
extern "C"{
#include <operator/transpose.h>
}
#undef restrict

SKYPAT_F(Operator_Acos, non_broadcast){
    // Prepare
    int32_t ndim = 3;
    int32_t dims[3] = {2,2,4};
    float A[16] = {0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15};
    float B[16];
    int32_t perm[3] = {1,0,2};
    float Ans[16] = {0,1,2,3,8,9,10,11,4,5,6,7,12,13,14,15};
    int32_t dataSize = 16;

    // Run
    ONNC_RUNTIME_transpose_float(NULL
        ,(const float*)A
        ,ndim,dims
        ,(float*)B
        ,ndim,dims
        ,perm
        ,ndim
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
