#include <skypat/skypat.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define restrict __restrict__
extern "C"{
#include <operator/sub.h>
}
#undef restrict

SKYPAT_F(Operator_sub, non_broadcast){
    // Prepare
    srand(time(NULL));
    int32_t ndim = rand() % 3 + 1;
    int32_t dims[ndim];
    int32_t dataSize = 1;
    for(int32_t i = 0; i < ndim; ++i){
        dims[i] = rand() % 6 + 1;
        dataSize *= dims[i];
    }
    float A[dataSize], B[dataSize], C[dataSize], Ans[dataSize];
    for(int32_t i = 0; i < dataSize; ++i){
        A[i] = rand() % 1000 / 100.0f;
        B[i] = rand() % 1000 / 100.0f;
		printf("Input : %f %f\n",A[i] ,B[i]);
        Ans[i] = A[i] - B[i];
    }
    // Run
    ONNC_RUNTIME_sub_float(NULL
        ,A
        ,ndim,dims
        ,B
        ,ndim,dims
		,C
		,ndim,dims
    );
    // Check
    for(int32_t i = 0; i < dataSize; ++i){
		printf("%f %f\n", C[i], Ans[i]);
        EXPECT_EQ(C[i], Ans[i]);
    }
}

int main(int argc, char *argv[]){
    skypat::Test::Initialize(argc, argv);
    skypat::Test::RunAll();
    return (skypat::testing::UnitTest::self()->getNumOfFails() == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
