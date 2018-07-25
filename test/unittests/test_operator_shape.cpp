#include <skypat/skypat.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define restrict __restrict__
extern "C"{
#include <operator/shape.h>
}
#undef restrict

SKYPAT_F(Operator_Shape, non_broadcast){
    // Prepare
    srand(time(NULL));
    int32_t ndim = rand() % 3 + 1;
    int32_t dims[ndim];
    int32_t dataSize = 1;
    for(int32_t i = 0; i < ndim; ++i){
        dims[i] = rand() % 100 + 1;
        dataSize *= dims[i];
    }
    float A[dataSize], B[ndim], Ans[ndim];
    for(int32_t i = 0; i < dataSize; ++i){
        A[i] = rand() % 1000 / 100.0f;
    }
    for(int32_t i = 0; i < ndim; ++i){
        Ans[i] = dims[i];
    }
    // Run
    ONNC_RUNTIME_shape_float(NULL
        ,A
        ,ndim,dims
        ,B
        ,1 , &ndim
    );
    // Check
    for(int32_t i = 0; i < ndim; ++i){
        EXPECT_EQ(B[i], Ans[i]);
    }
}

int main(int argc, char *argv[]){
    skypat::Test::Initialize(argc, argv);
    skypat::Test::RunAll();
    return (skypat::testing::UnitTest::self()->getNumOfFails() == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}