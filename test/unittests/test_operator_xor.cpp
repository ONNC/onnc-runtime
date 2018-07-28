#include <skypat/skypat.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define restrict __restrict__
extern "C"{
#include <operator/xor.h>
}
#undef restrict

SKYPAT_F(Operator_Xor, non_broadcast){
    // Prepare
    int32_t ndim_A = 4;
    int32_t dims_A[4] = {2,2,1,2};
    float A[8] = {0,1,1,0,1,1,0,0};

    int32_t ndim_B = 3;
    int32_t dims_B[3] = {2,4,1};
    float B[8] = {0,1,1,0,1,0,0,1};

    int32_t ndim_C = 4;
    int32_t dims_C[4] = {2,2,4,2};
    float C[32];

    //testing by numpy
    int32_t Ans[32] ={0,1,1,0,1,0,0,1,0,1,1,0,1,0,0,1,1,1,0,0,0,0,1,1,1,1,0,0,0,0,1,1};

    int32_t dataSize = 32;

    // Run
    ONNC_RUNTIME_xor_float(
    NULL,
    A,
    ndim_A,dims_A,
    B,
    ndim_B,dims_B,
    C,
    ndim_C,dims_C
  );
    // Check
    for(int32_t i = 0; i < dataSize; ++i){
        EXPECT_EQ((int32_t)C[i], Ans[i]);
    }
}

int main(int argc, char *argv[]){
    skypat::Test::Initialize(argc, argv);
    skypat::Test::RunAll();
    return (skypat::testing::UnitTest::self()->getNumOfFails() == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}