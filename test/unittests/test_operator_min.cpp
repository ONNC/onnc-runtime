#include <skypat/skypat.h>
#include <cstdlib>
#include <ctime>
#include <cmath>

#define restrict __restrict__
extern "C"{
#include <operator/min.h>
}
#undef restrict

SKYPAT_F(Operator_min, non_broadcast){
    // Prepare
    srand(time(NULL));
	int n = 4, ten_len = 6;
	const float ** input_data_0 = new const float * [n]{
		new const float [ten_len]{8, 1, 6, 8, 8, 8},
		new const float [ten_len]{100, 87, -1, 7, 7, 7},
		new const float [ten_len]{-87, 487, -5, 7, 1, 8},
		new const float [ten_len]{6, 6, 6, -2, 8, 7}
	};
	int input_data_0_ntensor = n;
	int input_data_0_ndim[] = {2, 2, 2, 2};	
	const int ** input_data_0_dims = new const int * [n]{
		new const int [2]{2, 3},
		new const int [2]{2, 3},
		new const int [2]{2, 3},
		new const int [2]{2, 3}
	};
	float output_min[6];
	int output_min_ndim = 2;
	const int output_min_dims[] = {2, 3};
	float Ans[ten_len] ;
	for(int j = 0 ; j < ten_len ; ++j){
		float mn = input_data_0[0][j];
		for(int i = 0 ; i < n ; ++i){
			mn = fmin(mn, input_data_0[i][j]);
		}
		Ans[j] = mn;
	}
    // Run
    ONNC_RUNTIME_min_float(NULL
        ,input_data_0
        ,input_data_0_ntensor
        ,input_data_0_ndim, input_data_0_dims
        ,output_min
		,output_min_ndim, output_min_dims
    );
    // Check
	for(int i = 0 ; i < ten_len ; ++i){
		printf("%f %f\n", Ans[i], output_min[i]);
		EXPECT_EQ(Ans[i], output_min[i]);
	}
}

int main(int argc, char *argv[]){
    skypat::Test::Initialize(argc, argv);
    skypat::Test::RunAll();
    return (skypat::testing::UnitTest::self()->getNumOfFails() == 0) ? EXIT_SUCCESS : EXIT_FAILURE;
}
