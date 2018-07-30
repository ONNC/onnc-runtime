#include <operator/pad.h>

#include <stdint.h>
#include <stdbool.h>
#include <string.h>
#define CONSTANT_MODE 0
#define EDGE_MODE 1
#define REFLECT_MODE 2


static inline void calculate_axis_dis(int32_t ndim, const int32_t * restrict dims, int32_t * restrict axis_dis){
    int32_t base = axis_dis[ndim-1] = 1;
    for(int32_t dim = ndim - 2 ; dim >= 0 ; --dim){
        axis_dis[dim] = base * dims[dim+1];
        base = axis_dis[dim];
    }
}

static inline bool next_dim(int32_t ndim, int32_t * restrict dim,
                            const int32_t * restrict dim_max) {
  do {
    ndim = ndim - 1;
    dim[ndim] += 1;
    if (dim[ndim] < dim_max[ndim]) {
      return true;
    } else { // reach dimension max
      if (ndim == 0) { // all dimension done
        return false;
      }
      dim[ndim] = 0;
    }
  } while(true);
}

static inline int64_t dim_to_offset(int32_t ndim, const int32_t * restrict dim,
                                    const int32_t * restrict axisDistance) {
  int64_t offset = 0;
  for (int32_t i = ndim - 1; i >= 0; --i) {
    offset += dim[i] * axisDistance[i];
  }
  return offset;
}

static inline void add_initial_to_output(
    const float * restrict input, int32_t input_ndim, const int32_t * restrict input_dims,
    int32_t * restrict axis_dis, int32_t * pads, int32_t * restrict axis_pad_dis,
    float * restrict output
){
    int32_t iter_index[input_ndim];
    int32_t fill_index[input_ndim];
    memset(iter_index, 0, sizeof(int32_t) * input_ndim);
    do{
        memcpy(fill_index, iter_index, sizeof(int32_t) * input_ndim);
        for(int32_t dim = 0 ; dim < input_ndim; dim++){
            fill_index[dim] += pads[dim];
        }

        output[dim_to_offset(input_ndim, fill_index, axis_pad_dis)]
        = input[dim_to_offset(input_ndim, iter_index, axis_dis)];

    }while(next_dim(input_ndim, iter_index, input_dims));
}

static bool in_obj_area(int32_t * restrict index, int32_t * restrict object, int32_t ndim){
    for(int32_t dim = 0 ; dim < ndim ; dim++){
        if(! (index[dim] >= object[dim] && index[dim] < object[dim + ndim])){
            return false;
        }
    }
    return true;
}

static inline void pad_along_axis(
    float * restrict output, int32_t output_ndim, const int32_t * restrict output_dims,
    const int32_t * restrict pads, const int32_t * restrict axis_pad_dis,
    int32_t * restrict object_area, const char * restrict mode, float value
){
    int32_t mode_no;
    if(strcmp(mode, "constant") == 0) mode_no = 0;
    else if(strcmp(mode, "edge") == 0) mode_no = 1;
    else if(strcmp(mode, "reflect") == 0) mode_no = 2;

    int32_t iter_index[output_ndim];
    memset(iter_index, 0, sizeof(int32_t) * output_ndim);

    switch(mode_no){
        case CONSTANT_MODE: {
            do{
                if(in_obj_area(iter_index, object_area, output_ndim)) continue;
                int32_t offset = dim_to_offset(output_ndim, iter_index, axis_pad_dis);
                output[offset] = value;
            }while(next_dim(output_ndim, iter_index, output_dims));
            break;
        }
        case EDGE_MODE: {
            break;
        }
        case REFLECT_MODE:{
            break;
        }
    }
}

void ONNC_RUNTIME_pad_float(
  void * restrict onnc_runtime_context
  ,const float * restrict input_data
  ,int32_t input_data_ndim, const int32_t * restrict input_data_dims
  ,float * restrict output_output
  ,int32_t output_output_ndim, const int32_t * restrict output_output_dims
  ,const char * restrict mode
  ,int32_t * restrict pads
  ,int32_t number_of_pads
  ,float value
) {
    // TODO: mode
    /* calculate axis_dis */
    int32_t axis_dis[input_data_ndim];
    calculate_axis_dis(input_data_ndim, input_data_dims, axis_dis);

    /* calculate input_pad_dims */
    int32_t input_pad_dims[input_data_ndim];
    memcpy(input_pad_dims, input_data_dims, sizeof(int32_t) * input_data_ndim);
    for(int32_t dim = 0 ; dim < input_data_ndim ; dim++){
        input_pad_dims[dim] += (pads[dim] + pads[dim + input_data_ndim]);
    }

    /* calculate axis_pad_dis */
    int32_t axis_pad_dis[input_data_ndim];
    calculate_axis_dis(input_data_ndim, input_pad_dims, axis_pad_dis);

    /* add initial value to output with new index */
    add_initial_to_output(input_data, input_data_ndim, input_data_dims, axis_dis,
                          pads, axis_pad_dis, output_output);
    /* initial object area */
    int32_t object_area[2*input_data_ndim];
    for(int32_t dim = 0 ; dim < input_data_ndim ; dim++){
        object_area[dim] = pads[dim];
        object_area[dim + input_data_ndim] = input_data_dims[dim] + pads[dim];
    }
    /* pad along with each axis */
    pad_along_axis(output_output, output_output_ndim, output_output_dims, pads, axis_pad_dis, object_area, mode, value);
}
