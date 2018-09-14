#include <operator/squeeze.h>

#include <stdio.h>
#include <stdint.h>
#include <stdbool.h>

void ONNC_RUNTIME_squeeze_float(
  void * restrict onnc_runtime_context
  ,const float * restrict input_data
  ,int32_t input_data_ndim, const int32_t * restrict input_data_dims
  ,float * restrict output_squeezed
  ,int32_t output_squeezed_ndim, const int32_t * restrict output_squeezed_dims
  ,int32_t * restrict axes
  ,int32_t number_of_axes
) {
    int32_t size_N = 1;

    // total counts of elements
    for(int32_t i = 0; i < input_data_ndim; ++i) {
        size_N *= input_data_dims[i];
    }

    for(int32_t i = 0; i < size_N; ++i) {
        output_squeezed[i] = input_data[i];
    }


    /*    
    // scan by axes or number_of_axes
    if((axes == NULL) || (number_of_axes == 0)) {
        int32_t j = 0;
        for(int32_t i = 0; i < input_data_ndim; ++i) {
            output_squeezed_dims[i] = 0;
            if(input_data_dims[i] != 1) {
                output_squeezed_dims[j] = input_data_dims[i];
                j++;
            }
        }
    } else {
        int32_t remove_index_counts = 0;
        // copy input_data dims array into output_data dims array
        for(int32_t i = 0; i < input_data_ndim; ++i) {
            output_squeezed_dims[i] = input_data_dims[i];
        }

        // compare dim's value to input dims array
        // if the value of the index is i, set the same index of output dim to -1  
        // just a mark
        for(int32_t i = 0; i < number_of_axes; ++i) {
            int32_t tmp_index = axes[i];
            if((tmp_index < output_squeezed_ndim) && (input_data_dims[tmp_index] == 1)) {
                output_squeezed_dims[tmp_index] = -1;
                remove_index_counts++;
            }
        }

        // search -1 from head to tail, if found -1, move the following data to
        // previous index.
        for(int32_t i = 0; i < output_squeezed_ndim; ++i) {
            if(output_squeezed_dims[i] == -1) {
                for(int32_t j = i; j < output_squeezed_ndim - 1; ++j) {
                    output_squeezed_dims[j] = output_squeezed_dims[j + 1];
                }
            }
        }

        // search -1 from head to tail, if found -1, move the following data to
        // previous index.
        for(int32_t i = output_squeezed_ndim - remove_index_counts; i < output_squeezed_ndim; ++i) {
            output_squeezed_dims[i] = 0;
        }
        output_squeezed_ndim -= remove_index_counts;
    }
    */
}

