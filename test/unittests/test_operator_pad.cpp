#include <skypat/skypat.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define restrict __restrict__
extern "C"{
#include <operator/pad.h>
}
#undef restrict

SKYPAT_F(Operator_Pad, non_broadcast){
    // Prepare
    srand(time(NULL));
    int32_t adim = 2, bdim = 2;
    int32_t adims[2]{3, 2}, bdims[2]{3, 4}, pads[4]{0, 2, 0, 0};
    float A[6]{
        1.0, 1.2, 2.3, 3.4, 4.5, 5.7
    };
    float B[12], Ans[12]{
        0.0, 0.0, 1.0, 1.2, 0.0, 0.0, 2.3, 3.4, 0.0, 0.0, 4.5, 5.7
    };
    // Run
    ONNC_RUNTIME_pad_float(NULL
        ,A
        ,adim,adims
        ,B
        ,bdim,bdims
        ,"constant"
        ,pads
        ,4
        ,0
    );
    // Check
    for(int32_t i = 0; i < 12; ++i){
        EXPECT_EQ(B[i], Ans[i]);
    }
}

int main(int argc, char *argv[]){
    skypat::Test::Initialize(argc, argv);
    skypat::Test::RunAll();
    return (skypat::testing::UnitTest::self()->getNumOfFails() == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
