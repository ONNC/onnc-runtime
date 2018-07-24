#include <skypat/skypat.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define restrict __restrict__
extern "C"{
#include <operator/argmax.h>
}
#undef restrict

SKYPAT_F(Operator_ArgMax, non_broadcast){
    // Prepare
    srand(time(NULL));
    int32_t ndim = 2;
    int32_t dims[2] = {
        3, 3
    };
    float A[9] = {
        2, 6, 3,
        5, 4, 1,
        8, 7, 9
    };
    float B[9];
    float Ans[9] = {
        1, 0, 2,
        0, 0, 0,
        0, 0, 0
    };
    // Run
    ONNC_RUNTIME_argmax_float(NULL
        ,A
        ,ndim,dims
        ,B
        ,ndim,dims
        ,1
        ,1
    );
    // Check
    for(int32_t i = 0; i < 3; ++i){
        EXPECT_EQ(B[i], Ans[i]);
    }
}

int main(int argc, char *argv[]){
    skypat::Test::Initialize(argc, argv);
    skypat::Test::RunAll();
    return (skypat::testing::UnitTest::self()->getNumOfFails() == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}