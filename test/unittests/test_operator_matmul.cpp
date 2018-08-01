#include <skypat/skypat.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define restrict __restrict__
extern "C"{
#include <operator/matmul.h>
}
#undef restrict

SKYPAT_F(Operator_matmul, non_broadcast){
    // Prepare
	int32_t dataSize = 1; 
	const float input_A[] = {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12};
	int32_t input_A_ndim = 3;
	const int32_t input_A_dims[] = {2, 2, 3};
	const float input_B[] = {13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24};
	int32_t input_B_ndim = 3;
	const int32_t input_B_dims[] = {2, 3, 2};
	float output_Y[1000];
	int32_t output_Y_ndim = 3;
	const int32_t output_Y_dims[] = {2, 2, 2} ;
	float Ans[] = {94, 100, 229, 244, 508, 532, 697, 730};
	for(int i = 0 ; i < output_Y_ndim ; ++i) dataSize *= output_Y_dims[i];
    // Run
    ONNC_RUNTIME_matmul_float(NULL,
		input_A,
		input_A_ndim, input_A_dims,
		input_B,
		input_B_ndim, input_B_dims,
		output_Y,
		output_Y_ndim, output_Y_dims
    );
    // Check
    for(int32_t i = 0; i < dataSize; ++i){
		printf("%f %f\n",output_Y[i], Ans[i]);
        EXPECT_EQ(output_Y[i], Ans[i]);
    }
}

int main(int argc, char *argv[]){
    skypat::Test::Initialize(argc, argv);
    skypat::Test::RunAll();
    return (skypat::testing::UnitTest::self()->getNumOfFails() == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
