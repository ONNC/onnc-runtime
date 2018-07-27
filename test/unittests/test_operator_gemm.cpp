#include <skypat/skypat.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define restrict __restrict__
extern "C"{
#include <operator/gemm.h>
}
#undef restrict

SKYPAT_F(Operator_Gemm, non_broadcast){
    // Prepare
    srand(time(NULL));
    int32_t ndim = 2;
    const int32_t *Adims = new const int32_t[2] {2, 3};
    const int32_t *Bdims = new const int32_t[2] {3, 2};
    const int32_t *Cdims = new const int32_t[2] {2, 2};
    const float *A = new float[6]{
        1, 2, 3,
        4, 5, 6
    };
    const float *B = new float[6]{
        1, 2,
        3, 4,
        5, 6
    };
    const float *C = new float[4]{
        2, 3,
        4, 5
    };
    float *Ans = new float[4]{
        22 * 3 + 4, 28 * 3 + 6,
        49 * 3 + 8, 64 * 3 + 10
    };
    float *Out = new float[4];
    // Run
    ONNC_RUNTIME_gemm_float(NULL
        ,A
        ,ndim,Adims
        ,B
        ,ndim,Bdims
        ,C
        ,ndim,Cdims
        ,Out
        ,ndim,Cdims
        ,3.0f
        ,2.0f
        ,0
        ,0
    );
    // Check
    for(int32_t i = 0; i < 4; ++i){
        EXPECT_EQ(Out[i], Ans[i]);
    }
}

SKYPAT_F(Operator_Gemm, non_broadcast_transAB){
    // Prepare
    srand(time(NULL));
    int32_t ndim = 2;
    const int32_t *Adims = new const int32_t[2] {2, 3};
    const int32_t *Bdims = new const int32_t[2] {3, 2};
    const int32_t *Cdims = new const int32_t[2] {3, 3};
    const float *A = new float[6]{
        1, 2, 3,
        4, 5, 6
    };
    const float *B = new float[6]{
        1, 2,
        3, 4,
        5, 6
    };
    const float *C = new float[9]{
        2, 3, 4,
        5, 6, 7,
        8, 9, 10
    };
    float *Ans = new float[9]{
         9 * 3 + 4,  19 * 3 + 6,  29 * 3 + 8,
        12 * 3 + 10, 26 * 3 + 12, 40 * 3 + 14,
        15 * 3 + 16, 33 * 3 + 18, 51 * 3 + 20,
    };
    float *Out = new float[9];
    // Run
    ONNC_RUNTIME_gemm_float(NULL
        ,A
        ,ndim,Adims
        ,B
        ,ndim,Bdims
        ,C
        ,ndim,Cdims
        ,Out
        ,ndim,Cdims
        ,3.0f
        ,2.0f
        ,1
        ,1
    );
    // Check
    for(int32_t i = 0; i < 9; ++i){
        EXPECT_EQ(Out[i], Ans[i]);
    }
}

int main(int argc, char *argv[]){
    skypat::Test::Initialize(argc, argv);
    skypat::Test::RunAll();
    return (skypat::testing::UnitTest::self()->getNumOfFails() == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
