#!/usr/bin/env python2
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from onnx import onnx

import sys
import struct

if __name__ == '__main__':
    if len(sys.argv) >= 2:
        filename = sys.argv[1]
    else:
        filename = 'input.pb'

    tensor = onnx.TensorProto()
    with open(filename, 'rb') as f:
        tensor.ParseFromString(f.read())

    assert tensor.data_type == 1

    ndim = len(tensor.dims)
    size = 1
    for dim in tensor.dims:
        size *= dim

    tensor_array = struct.unpack('%sf' % size, tensor.raw_data)
    def tensor_pretty_printer(offset, dim_index):
        if dim_index == ndim - 1:
            sys.stdout.write('  ' * dim_index + '[')
            for index in range(tensor.dims[dim_index]):
                sys.stdout.write('%f ' % (tensor_array[offset + index]))
            print(']')
            return tensor.dims[dim_index]
        else:
            print('  ' * dim_index + '[')
            orig_offset = offset
            for index in range(tensor.dims[dim_index]):
                offset += tensor_pretty_printer(offset ,dim_index + 1)
            print('  ' * dim_index + ']')
            return orig_offset - offset
    tensor_pretty_printer(0, 0);
    #print('[')
    #for i, dim in enumerate(tensor.dims):
    #    if i == ndim - 1:
    #    else:
    #        print('[')
    #        print(']')
    #print(']')
    #tensor.
