#include <skypat/skypat.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define restrict __restrict__
extern "C"{
#include <operator/clip.h>
}
#undef restrict

SKYPAT_F(Operator_Clip, non_broadcast){
    // Prepare
    srand(time(NULL));
    int32_t ndim = rand() % 3 + 1;
    int32_t dims[ndim];
    int32_t dataSize = 1;
    for(int32_t i = 0; i < ndim; ++i){
        dims[i] = rand() % 100 + 1;
        dataSize *= dims[i];
    }
    float A[dataSize], B[dataSize], Ans[dataSize];
    float max = rand() % 10000 / 100, min = rand() % 10000 / 100;
    if(min > max){
        float t = max;
        max = min;
        min = t;
    }
    for(int32_t i = 0; i < dataSize; ++i){
        A[i] = rand() % 10000 / 100;
        Ans[i] = (A[i] > max) ? max : (A[i] < min) ? min : A[i];
    }
    // Run
    ONNC_RUNTIME_clip_float(NULL
        ,A
        ,ndim,dims
        ,B
        ,ndim,dims
        ,max, min
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
