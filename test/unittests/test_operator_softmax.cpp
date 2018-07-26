#include <skypat/skypat.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define restrict __restrict__
extern "C"{
#include <operator/softmax.h>
}
#undef restrict

SKYPAT_F(Operator_Softmax, non_broadcast){
    // Prepare
    srand(time(NULL));
    int32_t ndim = rand() % 3 + 1;
    int32_t axis = rand() % ndim + 1;
    int32_t dims[ndim];
    int32_t dataSize = 1;
    int32_t nSize = 1, dSize = 1;
    for(int32_t i = 0; i < ndim; ++i){
        dims[i] = rand() % 100 + 1;
        dataSize *= dims[i];
    }
    for(int32_t i = 0; i < axis; ++i){
        nSize *= dims[i];
    }
    for(int32_t i = axis; i < ndim; ++i){
        dSize *= dims[i];
    }
    float A[dataSize], B[dataSize], Ans[dataSize];
    for(int32_t i = 0; i < dataSize; ++i){
        A[i] = (rand() % 1000) / 100.0f;
    }
    for(int32_t iN = 0; iN < nSize; ++iN){
        float* pInput = A + iN * dSize;
        float* pOutput = Ans + iN * dSize;
        float maxData = -__FLT_MAX__, sumData = 0.0f;
        for(int32_t i = 0; i < dSize; ++i){
            maxData = fmaxf(maxData, pInput[i]);
        }
        for(int32_t i = 0; i < dSize; ++i){
            pOutput[i] = expf(pInput[i] - maxData);
            sumData += pOutput[i];
        }
        for(int32_t i = 0; i < dSize; ++i){
            pOutput[i] /= sumData;
        }
    }

    // Run
    ONNC_RUNTIME_softmax_float(NULL
        ,A
        ,ndim,dims
        ,B
        ,ndim,dims
        ,axis
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
