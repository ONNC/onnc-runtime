#include <skypat/skypat.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define restrict __restrict__
extern "C"{
#include <operator/sum.h>
}
#undef restrict

SKYPAT_F(Operator_Sum, non_broadcast){
    // Prepare
    srand(time(NULL));
    int32_t ntensor = 3;
    int32_t *ndim = new int32_t[3]{
        2,2,2
    };
    const int32_t **dims = new const int32_t* [3]{
        new const int32_t[2] {2,2},
        new const int32_t[2] {2,2},
        new const int32_t[2] {2,2}
    };
    const float **A = new const float *[3]{
        new const float[4] {1, 2, 3, 4},
        new const float[4] {1, 2, 3, 4},
        new const float[4] {1, 2, 3, 4}
    };
    float B[4], Ans[4] = {3, 6, 9, 12};
    // Run
    ONNC_RUNTIME_sum_float(NULL
        ,A
        ,ntensor
        ,ndim,dims
        ,B
        ,ndim[0],dims[0]
    );
    // Check
    for(int32_t i = 0; i < 4; ++i){
        EXPECT_EQ(B[i], Ans[i]);
    }
}

int main(int argc, char *argv[]){
    skypat::Test::Initialize(argc, argv);
    skypat::Test::RunAll();
    return (skypat::testing::UnitTest::self()->getNumOfFails() == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
