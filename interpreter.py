#!/usr/bin/env python2
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx import onnx, helper, shape_inference
import sys
import struct

from ctypes import *
import os

def sizeof_tensor_datatype(data_type):
    return {
        1: 4,
        7: 8,
    }[data_type]

def parse_attribute(attributes):
    attrs = {}
    for attribute in attributes:
        attrs[attribute.name] = helper.get_attribute_value(attribute)
    print('%s' % attrs)
    return attrs

def typeproto_to_ndim_and_dims(type_proto):
    ndim = len(type_proto.tensor_type.shape.dim)
    dims = (c_int32 * ndim)()
    for idx, dim in enumerate(type_proto.tensor_type.shape.dim):
        dims[idx] = dim.dim_value
    return (ndim, cast(dims, POINTER(c_int32)))

def list_to_c_int32_array(pylist):
    return cast((c_int32 * len(pylist))(*pylist), POINTER(c_int32))

if __name__ == '__main__':
    assert len(sys.argv) >= 3

    model_filename = sys.argv[1]
    input_filename = sys.argv[2]

    onnx_model = onnx.load(model_filename)
    inferred_model = shape_inference.infer_shapes(onnx_model)

    MAGIC = ".TSR\0\0\0\0"
    offset = 0
    offset_table = []
    # Write weight.onnc.bin
    with open('weight.onnc.bin','wb') as weight_f:
        weight_f.write(MAGIC)
        offset += len(MAGIC)
        weight_f.write(struct.pack("<Q", len(inferred_model.graph.initializer)))
        offset += 8
        offset += 8 * 2 * len(inferred_model.graph.initializer)
        # Prepare offset table
        for tensor in inferred_model.graph.initializer:
            size = sizeof_tensor_datatype(tensor.data_type)
            for dim in tensor.dims:
                size *= dim
            if (size != len(tensor.raw_data)):
                print("size != len(tensor.raw_data)")
                sys.exit(1)
            offset_struct = (offset, size)
            weight_f.write(struct.pack("<QQ", *offset_struct))
            offset_table.append(offset_struct)
            offset += size
        # Write raw_data
        for tensor in inferred_model.graph.initializer:
            weight_f.write(tensor.raw_data)


    filepath = os.path.dirname(os.path.abspath(__file__))
    onnc_runtime = cdll.LoadLibrary(os.path.join(filepath, "libonnc-runtime.so"))

    onnc_runtime.ONNC_RUNTIME_init_runtime.restype = c_void_p
    onnc_runtime.ONNC_RUNTIME_init_runtime.argtypes = [c_char_p]
    #void *ONNC_RUNTIME_init_runtime(const char *onnx_model_file_name);

    onnc_runtime.ONNC_RUNTIME_shutdown_runtime.restype = c_bool
    onnc_runtime.ONNC_RUNTIME_shutdown_runtime.argtypes = [c_void_p]
    #bool ONNC_RUNTIME_shutdown_runtime(void *onnc_runtime_context);

    onnc_runtime.ONNC_RUNTIME_load_weight.restype = c_void_p
    onnc_runtime.ONNC_RUNTIME_load_weight.argtypes = [c_void_p, c_uint32]
    #void *ONNC_RUNTIME_load_weight(void *onnc_runtime_context, uint32_t weight_index);

    onnc_runtime.ONNC_RUNTIME_internal_allocate_memory.restype = c_void_p
    onnc_runtime.ONNC_RUNTIME_internal_allocate_memory.argtypes = [c_void_p, c_size_t, c_size_t]
    #void *ONNC_RUNTIME_internal_allocate_memory(void *onnc_runtime_context, size_t num, size_t size);

    context = onnc_runtime.ONNC_RUNTIME_init_runtime("weight.onnc.bin")

    address_table = {}
    ndim_and_dims_table = {}

    # load weight
    for i, tensor in enumerate(inferred_model.graph.initializer):
        address_table[tensor.name] = onnc_runtime.ONNC_RUNTIME_load_weight(context, i)
        #print(tensor.name, ": ", hex(address_table[tensor.name]))

    # load input
    only_one_input = True
    for value_info in inferred_model.graph.input:
        ndim_and_dims_table[value_info.name] = typeproto_to_ndim_and_dims(value_info.type)
        if not (value_info.name in address_table):
            assert only_one_input

            tensor = onnx.TensorProto()
            with open(input_filename, 'rb') as f:
                tensor.ParseFromString(f.read())
            assert tensor.data_type == 1

            ndim = len(tensor.dims)
            size = 1
            for dim in tensor.dims:
                size *= dim

            input_array = (c_float * size).from_buffer_copy(tensor.raw_data)
            address_table[value_info.name] = cast(input_array, POINTER(c_float))
            only_one_input = False

    # allocate output memory
    def allocate_output_memory(value_info):
        tensor_type = value_info.type.tensor_type
        size = 1
        for dim in tensor_type.shape.dim:
            size *= dim.dim_value
        address_table[value_info.name] = \
            onnc_runtime.ONNC_RUNTIME_internal_allocate_memory(context,
                                                               size,
                                                               sizeof_tensor_datatype(tensor_type.elem_type))
        ndim_and_dims_table[value_info.name] = typeproto_to_ndim_and_dims(value_info.type)

    for value_info in inferred_model.graph.value_info:
        allocate_output_memory(value_info)
    for value_info in inferred_model.graph.output:
        allocate_output_memory(value_info)

    def passInput0ToOutput0(node):
        address_table[node.output[0]] = address_table[node.input[0]]

    def run_conv(node):
        attributes = parse_attribute(node.attribute)
        assert not ('auto_pad' in attributes)
        X = address_table[node.input[0]]
        W = address_table[node.input[1]]
        B = address_table[node.input[2]] if len(node.input) == 3 else None
        Y = address_table[node.output[0]]
        (ndim, X_dim) = ndim_and_dims_table[node.input[0]]
        (_, W_dim) = ndim_and_dims_table[node.input[1]]
        (_, Y_dim) = ndim_and_dims_table[node.output[0]]
        auto_pad = 0
        dilations = list_to_c_int32_array(attributes['dilations'] if 'dilations' in attributes else ([1] * (ndim - 2)))
        group = attributes['group'] if 'group' in attributes else 1
        kernel_shape = None # TODO
        pads = list_to_c_int32_array(attributes['pads'] if 'pads' in attributes else ([0] * (ndim - 2) * 2))
        strides = list_to_c_int32_array(attributes['strides'] if 'strides' in attributes else ([1] * (ndim - 2)))
        if ndim == 4:
            onnc_runtime.ONNC_RUNTIME_conv_2d_float(context,
                                                    X_dim[0], X_dim[1], X_dim[2], X_dim[3],
                                                    X,
                                                    W_dim[0], W_dim[1], W_dim[2], W_dim[3],
                                                    W,
                                                    B,
                                                    Y_dim[0], Y_dim[1], Y_dim[2], Y_dim[3],
                                                    Y,
                                                    auto_pad,
                                                    dilations,
                                                    group,
                                                    kernel_shape,
                                                    pads,
                                                    strides)
        else:
            onnc_runtime.ONNC_RUNTIME_conv_float(context,
                                                 X, W,
                                                 ndim, X_dim,
                                                 W_dim,
                                                 B, Y,
                                                 Y_dim,
                                                 auto_pad,
                                                 dilations,
                                                 group,
                                                 kernel_shape,
                                                 pads,
                                                 strides)
    onnc_runtime.ONNC_RUNTIME_conv_float.restype = None
    onnc_runtime.ONNC_RUNTIME_conv_float.argtypes = [c_void_p,
                                                     c_void_p, c_void_p,
                                                     c_int32, POINTER(c_int32),
                                                     POINTER(c_int32),
                                                     c_void_p, c_void_p,
                                                     POINTER(c_int32),
                                                     c_int32,
                                                     POINTER(c_int32),
                                                     c_int32,
                                                     POINTER(c_int32),
                                                     POINTER(c_int32),
                                                     POINTER(c_int32)]
    #void ONNC_RUNTIME_conv_float(void * restrict onnc_runtime_context,
    #                             const float * restrict X, const float * restrict W,
    #                             int32_t ndim, const int32_t * restrict X_dim,
    #                             const int32_t * restrict W_dim,
    #                             const float * restrict B, float * restrict Y,
    #                             const int32_t * restrict Y_dim,
    #                             int32_t auto_pad,
    #                             const int32_t * restrict dilations,
    #                             int32_t group,
    #                             const int32_t * restrict kernel_shape,
    #                             const int32_t * restrict pads,
    #                             const int32_t * restrict strides);
    onnc_runtime.ONNC_RUNTIME_conv_2d_float.restype = None
    onnc_runtime.ONNC_RUNTIME_conv_2d_float.argtypes = [c_void_p,
                                                        c_int32, c_int32, c_int32, c_int32,
                                                        c_void_p,
                                                        c_int32, c_int32, c_int32, c_int32,
                                                        c_void_p,
                                                        c_void_p,
                                                        c_int32, c_int32, c_int32, c_int32,
                                                        c_void_p,
                                                        c_int32,
                                                        POINTER(c_int32),
                                                        c_int32,
                                                        POINTER(c_int32),
                                                        POINTER(c_int32),
                                                        POINTER(c_int32)]
    #void ONNC_RUNTIME_conv_2d_float(void * restrict onnc_runtime_context,
    #                                int32_t N, int32_t C, int32_t iH, int32_t iW,
    #                                const float X[restrict N][C][iH][iW],
    #                                int32_t M, int32_t kC, int32_t kH, int32_t kW,
    #                                const float W[restrict M][kC][kH][kW],
    #                                const float B[restrict M],
    #                                int32_t oN, int32_t oC, int32_t oH, int32_t oW,
    #                                float Y[restrict oN][oC][oH][oW],
    #                                int32_t auto_pad,
    #                                const int32_t * restrict dilations,
    #                                int32_t group,
    #                                const int32_t * restrict kernel_shape,
    #                                const int32_t * restrict pads,
    #                                const int32_t * restrict strides);

    def run_gemm(node):
        attributes = parse_attribute(node.attribute)
        A = address_table[node.input[0]]
        (ndim, A_dim) = ndim_and_dims_table[node.input[0]]
        B = address_table[node.input[1]]
        C = address_table[node.input[2]]
        (ncdim, C_dim) = ndim_and_dims_table[node.input[2]]
        Y = address_table[node.output[0]]
        (nydim, Y_dim) = ndim_and_dims_table[node.output[0]]
        M = A_dim[0]
        K = A_dim[1]
        N = Y_dim[1]
        alpha = attributes['alpha'] if 'alpha' in attributes else 1.0
        beta = attributes['beta'] if 'beta' in attributes else 1.0
        broadcast = attributes['broadcast'] if 'broadcast' in attributes else 0
        transA = attributes['transA'] if 'transA' in attributes else 0
        transB = attributes['transB'] if 'transB' in attributes else 0
        onnc_runtime.ONNC_RUNTIME_gemm_float(context,
                                             A,
                                             B,
                                             M, K, N,
                                             C,
                                             ncdim, C_dim,
                                             Y,
                                             nydim, Y_dim,
                                             alpha,
                                             beta,
                                             broadcast,
                                             transA,
                                             transB)
    onnc_runtime.ONNC_RUNTIME_gemm_float.restype = None
    onnc_runtime.ONNC_RUNTIME_gemm_float.argtypes = [c_void_p,
                                                     c_void_p,
                                                     c_void_p,
                                                     c_int32, c_int32, c_int32,
                                                     c_void_p,
                                                     c_int32, POINTER(c_int32),
                                                     c_void_p,
                                                     c_int32, POINTER(c_int32),
                                                     c_float,
                                                     c_float,
                                                     c_int32,
                                                     c_int32,
                                                     c_int32]
    #void ONNC_RUNTIME_gemm_float(void * restrict onnc_runtime_context,
    #                             const float * restrict A,
    #                             const float * restrict B,
    #                             int32_t M, int32_t K, int32_t N,
    #                             const float * restrict C,
    #                             int32_t ncdim, const int32_t * restrict C_dim,
    #                             float * restrict Y,
    #                             int32_t nydim, const int32_t * restrict Y_dim,
    #                             float alpha,
    #                             float beta,
    #                             int32_t broadcast,
    #                             int32_t transA,
    #                             int32_t transB);

    def run_maxpool(node):
        attributes = parse_attribute(node.attribute)
        assert not ('auto_pad' in attributes)
        X = address_table[node.input[0]]
        (ndim, X_dim) = ndim_and_dims_table[node.input[0]]
        Y = address_table[node.output[0]]
        (_, Y_dim) = ndim_and_dims_table[node.output[0]]
        auto_pad = 0
        kernel_shape = list_to_c_int32_array(attributes['kernel_shape'])
        pads = list_to_c_int32_array(attributes['pads'] if 'pads' in attributes else ([0] * (ndim - 2) * 2))
        strides = list_to_c_int32_array(attributes['strides'] if 'strides' in attributes else ([1] * (ndim - 2)))
        onnc_runtime.ONNC_RUNTIME_maxpool_float(context,
                                                X,
                                                ndim, X_dim,
                                                Y,
                                                Y_dim,
                                                auto_pad,
                                                kernel_shape,
                                                pads,
                                                strides)
    onnc_runtime.ONNC_RUNTIME_maxpool_float.restype = None
    onnc_runtime.ONNC_RUNTIME_maxpool_float.argtypes = [c_void_p,
                                                        c_void_p,
                                                        c_int32, POINTER(c_int32),
                                                        c_void_p,
                                                        POINTER(c_int32),
                                                        c_int32,
                                                        POINTER(c_int32),
                                                        POINTER(c_int32),
                                                        POINTER(c_int32)]
    #void ONNC_RUNTIME_maxpool_float(void * restrict onnc_runtime_context,
    #                                const float * restrict X,
    #                                int32_t ndim, const int32_t * restrict X_dim,
    #                                float * restrict Y,
    #                                const int32_t * restrict Y_dim,
    #                                int32_t auto_pad,
    #                                const int32_t * restrict kernel_shape,
    #                                const int32_t * restrict pads,
    #                                const int32_t * restrict strides);

    def run_relu(node):
        X = address_table[node.input[0]]
        (ndim, X_dim) = ndim_and_dims_table[node.input[0]]
        Y = address_table[node.output[0]]
        onnc_runtime.ONNC_RUNTIME_relu_float(context,
                                             X,
                                             ndim, X_dim,
                                             Y)
    onnc_runtime.ONNC_RUNTIME_relu_float.restype = None
    onnc_runtime.ONNC_RUNTIME_relu_float.argtypes = [c_void_p,
                                                     c_void_p,
                                                     c_int32, POINTER(c_int32),
                                                     c_void_p]
    #void ONNC_RUNTIME_relu_float(void * restrict onnc_runtime_context,
    #                             const float * restrict X,
    #                             int32_t ndim, const int32_t * restrict X_dim,
    #                             float * restrict Y);

    def run_softmax(node):
        attributes = parse_attribute(node.attribute)
        input = address_table[node.input[0]]
        (ndim, input_dim) = ndim_and_dims_table[node.input[0]]
        output = address_table[node.output[0]]
        axis = attributes['axis'] if 'axis' in attributes else 1
        onnc_runtime.ONNC_RUNTIME_softmax_float(context,
                                                input,
                                                ndim, input_dim,
                                                axis,
                                                output)
    onnc_runtime.ONNC_RUNTIME_softmax_float.restype = None
    onnc_runtime.ONNC_RUNTIME_softmax_float.argtypes = [c_void_p,
                                                        c_void_p,
                                                        c_int32, POINTER(c_int32),
                                                        c_int32,
                                                        c_void_p]
#void ONNC_RUNTIME_softmax_float(void * restrict onnc_runtime_context,
#                                const float * restrict input,
#                                int32_t ndim, const int32_t * restrict input_dim,
#                                int32_t axis,
#                                float * restrict output);

    def run_lrn(node):
        attributes = parse_attribute(node.attribute)
        X = address_table[node.input[0]]
        (ndim, X_dim) = ndim_and_dims_table[node.input[0]]
        Y = address_table[node.output[0]]
        alpha = attributes['alpha']
        beta = attributes['beta']
        bias = attributes['bias'] if 'bias' in attributes else 1.0
        size = attributes['size']
        onnc_runtime.ONNC_RUNTIME_lrn_float(context,
                                            X,
                                            ndim, X_dim,
                                            alpha,
                                            beta,
                                            bias,
                                            size,
                                            Y)
    onnc_runtime.ONNC_RUNTIME_lrn_float.restype = None
    onnc_runtime.ONNC_RUNTIME_lrn_float.argtypes = [c_void_p,
                                                    c_void_p,
                                                    c_int32, POINTER(c_int32),
                                                    c_float,
                                                    c_float,
                                                    c_float,
                                                    c_int32,
                                                    c_void_p]
    #void ONNC_RUNTIME_LRN_float(void * restrict onnc_runtime_context,
    #                            const float * restrict X,
    #                            int32_t ndim, const int32_t * restrict X_dim,
    #                            float alpha,
    #                            float beta,
    #                            float bias,
    #                            int32_t size,
    #                            float * restrict Y);

    def run_reshape(node):
        # XXX: HACK
        (ndim, X_dim) = ndim_and_dims_table[node.input[0]]
        X_dim[1] *= X_dim[2]
        X_dim[1] *= X_dim[3]
        ndim_and_dims_table[node.output[0]] = (ndim - 2, X_dim)
        address_table[node.output[0]] = address_table[node.input[0]]

    def run_batchnormalization(node):
        attributes = parse_attribute(node.attribute)
        X = address_table[node.input[0]]
        scale = address_table[node.input[1]]
        B = address_table[node.input[2]]
        inMean = address_table[node.input[3]]
        inVar = address_table[node.input[4]]
        (ndim, X_dim) = ndim_and_dims_table[node.input[0]]
        Y = address_table[node.output[0]]
        outMean = address_table[node.output[1]] if (len(node.output) > 1) else None
        outVar = address_table[node.output[2]] if (len(node.output) > 2) else None
        savedMean = address_table[node.output[3]] if (len(node.output) > 3) else None
        savedVar = address_table[node.output[4]] if (len(node.output) > 4) else None
        eps = attributes['epsilon'] if 'epsilon' in attributes else 1e-5
        momentum = attributes['momentum'] if 'momentum' in attributes else 0.9
        spatial = attributes['spatial'] if 'spatial' in attributes else 1
        onnc_runtime.ONNC_RUNTIME_batchnormalization_float(context,
                                                           X, ndim, X_dim,
                                                           Y,
                                                           scale,
                                                           B,
                                                           inMean,
                                                           inVar,
                                                           outMean,
                                                           outVar,
                                                           savedMean,
                                                           savedVar,
                                                           eps,
                                                           momentum,
                                                           spatial)
    onnc_runtime.ONNC_RUNTIME_batchnormalization_float.restype = None
    onnc_runtime.ONNC_RUNTIME_batchnormalization_float.argtypes = [c_void_p,
                                                    c_void_p,
                                                    c_int32, POINTER(c_int32),
                                                    c_void_p,
                                                    c_void_p,
                                                    c_void_p,
                                                    c_void_p,
                                                    c_void_p,
                                                    c_void_p,
                                                    c_void_p,
                                                    c_void_p,
                                                    c_void_p,
                                                    c_float,
                                                    c_float,
                                                    c_int32]
    # void ONNC_RUNTIME_batchnormalization_float(void * restrict onnc_runtime_context,
    #                                            const float * restrict X,
    #                                            int32_t ndim, const int32_t * restrict X_dim,
    #                                            float * restrict Y,
    #                                            const float * restrict scale,
    #                                            const float * restrict B,
    #                                            const float * restrict meanI,
    #                                            const float * restrict varI,
    #                                            float * restrict meanO,
    #                                            float * restrict varO,
    #                                            float * restrict saved_mean,
    #                                            float * restrict saved_var,
    #                                            float epsilon,
    #                                            float momentum,
    #                                            int32_t spatial);

    def run_add(node):
        attributes = parse_attribute(node.attribute)
        A = address_table[node.input[0]]
        B = address_table[node.input[1]]
        (ndim, A_dim) = ndim_and_dims_table[node.input[0]]
        C = address_table[node.output[0]]
        onnc_runtime.ONNC_RUNTIME_add_float(context,
                                            A,
                                            ndim, A_dim,
                                            B,
                                            C)
    onnc_runtime.ONNC_RUNTIME_add_float.restype = None
    onnc_runtime.ONNC_RUNTIME_add_float.argtypes = [c_void_p,
                                                    c_void_p,
                                                    c_int32, POINTER(c_int32),
                                                    c_void_p,
                                                    c_void_p]
    # void ONNC_RUNTIME_add_float(void * restrict onnc_runtime_context,
    #                             const float * restrict A,
    #                             int32_t ndim, const int32_t * restrict A_dim,
    #                             const float * restrict B,
    #                             float * restrict C);

    def run_averagepool(node):
        attributes = parse_attribute(node.attribute)
        X = address_table[node.input[0]]
        (ndim, X_dim) = ndim_and_dims_table[node.input[0]]
        Y = address_table[node.output[0]]
        (_, Y_dim) = ndim_and_dims_table[node.output[0]]
        auto_pad = attributes['auto_pad'] if 'auto_pad' in attributes else 0
        count_include_pad = attributes['count_include_pad'] if 'auto_pad' in attributes else 0
        kernel_shape = list_to_c_int32_array(attributes['kernel_shape'])
        pads = list_to_c_int32_array(attributes['pads'] if 'pads' in attributes else ([0] * (ndim - 2) * 2))
        strides = list_to_c_int32_array(attributes['strides'] if 'strides' in attributes else ([1] * (ndim - 2)))
        onnc_runtime.ONNC_RUNTIME_averagepool_float(context,
                                            X,
                                            ndim, X_dim,
                                            Y, Y_dim,
                                            auto_pad,
                                            count_include_pad,
                                            kernel_shape,
                                            pads,
                                            strides)
    onnc_runtime.ONNC_RUNTIME_averagepool_float.restype = None
    onnc_runtime.ONNC_RUNTIME_averagepool_float.argtypes = [c_void_p,
                                                    c_void_p,
                                                    c_int32, POINTER(c_int32),
                                                    c_void_p,
                                                    POINTER(c_int32),
                                                    c_int32,
                                                    c_int32,
                                                    POINTER(c_int32),
                                                    POINTER(c_int32),
                                                    POINTER(c_int32)]
    # void ONNC_RUNTIME_averagepool_float(void * restrict onnc_runtime_context,
    #                                     const float * restrict X,
    #                                     int32_t ndim, const int32_t * restrict X_dim,
    #                                     float * restrict Y,
    #                                     const int32_t * restrict Y_dim,
    #                                     int32_t auto_pad,
    #                                     int32_t count_include_pad,
    #                                     const int32_t * restrict kernel_shape,
    #                                     const int32_t * restrict pads,
    #                                     const int32_t * restrict strides); 
    for node in inferred_model.graph.node:
        {
            'Conv': run_conv,
            'Gemm': run_gemm,
            'MaxPool': run_maxpool,
            'Relu': run_relu,
            'Softmax': run_softmax,
            'Reshape': run_reshape,
            'Add': run_add,
            'LRN': run_lrn,
            'BatchNormalization': run_batchnormalization,
            'Dropout': passInput0ToOutput0,
            'Sum': run_add,
            'AveragePool': run_averagepool,
        }[node.op_type](node)
        out_array = cast(address_table[node.output[0]], POINTER(c_float))
        (ndim, dims) = ndim_and_dims_table[node.output[0]]
        out_sum = [0]
        def tensor_printer(offset, dim_index):
            if dim_index == ndim - 1:
                for index in range(dims[dim_index]):
                    out_sum[0] += out_array[offset + index]
                return dims[dim_index]
            else:
                orig_offset = offset
                for index in range(dims[dim_index]):
                    offset += tensor_printer(offset ,dim_index + 1)
                return orig_offset - offset
        tensor_printer(0, 0)
        if out_sum[0] == 0:
            sys.exit(-1)


    # print output
    for output in inferred_model.graph.output:
        (ndim, dims) = ndim_and_dims_table[output.name]
        size = 1
        for idx in range(ndim):
            size *= dims[idx]
        tensor_array = cast(address_table[output.name], POINTER(c_float))

        def tensor_pretty_printer(offset, dim_index):
            if dim_index == ndim - 1:
                sys.stdout.write('  ' * dim_index + '[')
                for index in range(dims[dim_index]):
                    sys.stdout.write('%f ' % (tensor_array[offset + index]))
                print(']')
                return dims[dim_index]
            else:
                print('  ' * dim_index + '[')
                orig_offset = offset
                for index in range(dims[dim_index]):
                    offset += tensor_pretty_printer(offset ,dim_index + 1)
                print('  ' * dim_index + ']')
                return orig_offset - offset
        tensor_pretty_printer(0, 0)

    succ = onnc_runtime.ONNC_RUNTIME_shutdown_runtime(context)
    #print(succ)
