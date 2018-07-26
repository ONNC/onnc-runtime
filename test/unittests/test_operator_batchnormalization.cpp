#include <skypat/skypat.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define restrict __restrict__
extern "C"{
#include <operator/batchnormalization.h>
}
#undef restrict

SKYPAT_F(Operator_Batchnormalization, non_broadcast){
    // Prepare
    srand(time(NULL));
    int32_t dims[4] = {
        3, 2, 1, 3
    };
    float A[18], B[18], Ans[18];
    for(int32_t i = 0; i < 18; ++i){
        A[i] = rand() % 100 / 10.0f;
    }
    float scale[2] = {
        1.0f, 1.0f
    };
    int32_t sdims[1] = {
        2
    };
    float bias[2] = {
        0.0f, 0.0f
    };
    float mean[6], var[6];
    for(int32_t i = 0; i < 6; ++i){
        mean[i] = rand() % 100 / 10.0f;
        var[i] = rand() % 100 / 10.0f;
    }
    float epsilon = 1e-5f;
    for(int32_t n = 0; n < 3; ++n){
        for(int32_t c = 0; c < 2; ++c){
            float *pA = A + n * 6 + c * 3;
            float *pAns = Ans + n * 6 + c * 3;
            float *pMean = mean + n * 2;
            float *pVar = var + n * 2;
            for(int32_t i = 0; i < 3; ++i){
                pAns[i] = scale[c] * (pA[i] - pMean[c]) / sqrtf(pVar[c] + epsilon) + bias[c];
            }
        }
    }
    // Run
    ONNC_RUNTIME_batchnormalization_float(NULL
        ,A, 4, dims
        ,scale ,1 ,sdims
        ,bias ,1 ,sdims
        ,mean ,1 ,sdims
        ,var ,1 ,sdims
        ,B ,4, dims
        ,mean ,1 ,sdims
        ,var ,1 ,sdims
        ,mean ,1 ,sdims
        ,var ,1 ,sdims
        ,epsilon
        ,0.9
        ,1
    );
    // Check
    for(int32_t i = 0; i < 18; ++i){
        EXPECT_EQ(B[i], Ans[i]);
    }
}

int main(int argc, char *argv[]){
    skypat::Test::Initialize(argc, argv);
    skypat::Test::RunAll();
    return (skypat::testing::UnitTest::self()->getNumOfFails() == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
