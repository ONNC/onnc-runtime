#include <skypat/skypat.h>
#include <cstdlib>
#include <cmath>

#define restrict __restrict__
extern "C"{
#include <operator/mean.h>
}
#undef restrict

SKYPAT_F(Operator_Mean, non_broadcast){
    // Prepare
    int32_t ndimArr[3] = {
        2, 2, 2
    };
    const int32_t **dims = new const int32_t *[3];
    for(int i = 0; i < 3; ++i){
        dims[i] = new const int32_t[2]{
            2, 3
        };
    }

    float **A = new float *[3]{
        new float[6]{2,3,4,5,6,7},
        new float[6]{3,4,5,6,7,8},
        new float[6]{4,5,6,7,8,9}
    };
    float B[6], Ans[6] = {3,4,5,6,7,8};
    
    // Run
    ONNC_RUNTIME_mean_float(NULL
        ,(const float **)A ,3
        ,ndimArr, dims
        ,B
        ,3, dims[0]
    );
    // Check
    for(int32_t i = 0; i < 6; ++i){
        EXPECT_EQ(B[i], Ans[i]);
    }
}

int main(int argc, char *argv[]){
    skypat::Test::Initialize(argc, argv);
    skypat::Test::RunAll();
    return (skypat::testing::UnitTest::self()->getNumOfFails() == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
