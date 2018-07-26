#include <skypat/skypat.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define restrict __restrict__
extern "C"{
#include <operator/neg.h>
}
#undef restrict

SKYPAT_F(Operator_Neg, non_broadcast){
    // Prepare
    srand(time(NULL));
    int32_t ndim = 2;
    int32_t dims[2] = {
        3, 3
    };
    float X[dims[0]][dims[1]] = {0};
    float Y[dims[0]][dims[1]];
    float testAns[dims[0]][dims[1]] = {0};
    for (int32_t i = 0; i < dims[0]; i++){
        for (int32_t j = 0; j < dims[1]; j++){
            X[i][j] = rand();
            testAns[i][j] = -X[i][j];
        }
    }
    
    // Run
    ONNC_RUNTIME_neg_float(NULL
        ,(const float*)X
        ,ndim,dims
        ,(float*)Y
        ,ndim,dims
    );
    // Check
    for (int32_t i = 0; i < dims[0]; i++){
        for (int32_t j = 0; j < dims[1]; j++){
            EXPECT_EQ(Y[i][j], testAns[i][j]);
        }
    }
}

int main(int argc, char *argv[]){
    skypat::Test::Initialize(argc, argv);
    skypat::Test::RunAll();
    return (skypat::testing::UnitTest::self()->getNumOfFails() == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}