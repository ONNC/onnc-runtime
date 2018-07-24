#include <skypat/skypat.h>
#include <cstdlib>
#include <ctime>

#define restrict __restrict__
extern "C"{
#include <operator/equal.h>
}
#undef restrict

SKYPAT_F(Operator_Equal, non_broadcast){
    // Prepare
    srand(time(NULL));
    int32_t ndim = rand() % 3 + 1;
    int32_t dims[ndim];
    int32_t dataSize = 1;
    for(int32_t i = 0; i < ndim; ++i){
        dims[i] = rand() % 100 + 1;
        dataSize *= dims[i];
    }
    float A[dataSize], B[dataSize], C[dataSize], Ans[dataSize];
    for(int32_t i = 0; i < dataSize; ++i){
        A[i] = rand() % 1000 / 100.0f;
        B[i] = rand() % 1000 / 100.0f;
        Ans[i] = A[i] == B[i];
    }
    // Run
    ONNC_RUNTIME_equal_float(NULL
        ,A
        ,ndim,dims
        ,B
        ,ndim,dims
        ,C
        ,ndim,dims
    );
    // Check
    for(int32_t i = 0; i < dataSize; ++i){
        EXPECT_EQ(C[i], Ans[i]);
    }
}

int main(int argc, char *argv[]){
    skypat::Test::Initialize(argc, argv);
    skypat::Test::RunAll();
    return (skypat::testing::UnitTest::self()->getNumOfFails() == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}