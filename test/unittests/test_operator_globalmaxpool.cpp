#include <skypat/skypat.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define restrict __restrict__
extern "C"{
#include <operator/globalmaxpool.h>
}
#undef restrict

SKYPAT_F(Operator_globalmaxpool, non_broadcast){
    // Prepare
	int32_t dataSize = 1 ;
	const float input_X[] = {1, 2, 3, -100, 666, 8, 9, 7};
	int32_t input_X_ndim = 3;
	const int32_t input_X_dims[] = {2,2,2};
	float output_Y[1000];
	int32_t output_Y_ndim = 1;
	const int32_t output_Y_dims[] = {1};
	
	float Ans[] = {666};
    // Run
    ONNC_RUNTIME_globalmaxpool_float(NULL,
		input_X,
		input_X_ndim, input_X_dims,
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
