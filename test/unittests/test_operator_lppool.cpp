#include <skypat/skypat.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define restrict __restrict__
extern "C"{
#include <operator/lppool.h>
}
#undef restrict

SKYPAT_F(Operator_lppool, non_broadcast){
    // Prepare
	const float input_X[] = {3, -1, 8, 2, 7, -5, 9, 8, 7, 8, 7, -1, 7, 6, 7, 10, 9, 100};
	int32_t input_X_ndim = 4;
	const int32_t input_X_dims[] = {2, 1, 3, 3};
	float output_Y[1000];
	int32_t output_Y_ndim = 4;
	const int32_t output_Y_dims[] = {2, 1, 3, 2};
	const char auto_pad[] = "HIIIIII";
	int32_t kernel_shape[] = {2, 3};
	int32_t number_of_kernel_shape = 2;
	int32_t p = 3 ;
	int32_t pads[] = {1, 1, 0, 2};
	int32_t number_of_pads = 4;
	int32_t strides[] = {1, 3};
	int32_t number_of_strides = 2;

	int32_t dataSize = 1 ;
	for(int i = 0 ; i < output_Y_ndim ; ++i) dataSize *= output_Y_dims[i];

	float Ans[] = {3.036588, 8, 7.236797, 8.604252, 11.676544, 7.763936, 
				   9.491219, 1, 11.224055, 7.006796, 13.177064, 100.011432};
    // Run
    ONNC_RUNTIME_lppool_float(NULL,
		input_X,
		input_X_ndim, input_X_dims,
		output_Y,
		output_Y_ndim, output_Y_dims,
		auto_pad,
		kernel_shape,
		number_of_kernel_shape,
		p,
		pads,
		number_of_pads,
		strides,
		number_of_strides
    );
    // Check
    for(int32_t i = 0; i < dataSize; ++i){
		printf("%f %f\n",output_Y[i], Ans[i]);
        EXPECT_TRUE( fabs ( output_Y[i] - Ans[i] ) < 1e-4 );
    }
}

int main(int argc, char *argv[]){
    skypat::Test::Initialize(argc, argv);
    skypat::Test::RunAll();
    return (skypat::testing::UnitTest::self()->getNumOfFails() == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
