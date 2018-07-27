//===- Interpreter.cpp ----------------------------------------------------===//
//
//                             The ONNC Project
//
// See LICENSE.TXT for details.
//
//===----------------------------------------------------------------------===//
#include "Interpreter.h"
#include <onnc/Support/IOStream.h>

using namespace onnc;


size_t sizeof_tensor_type(const ::onnx::TensorProto_DataType& dataType){
  switch(dataType){
    case ::onnx::TensorProto_DataType_BOOL:
      return sizeof(bool);
    case ::onnx::TensorProto_DataType_INT8:
    case ::onnx::TensorProto_DataType_UINT8:
      return 1;
    case ::onnx::TensorProto_DataType_UINT16:
    case ::onnx::TensorProto_DataType_INT16:
    case ::onnx::TensorProto_DataType_FLOAT16:
      return 2;
    case ::onnx::TensorProto_DataType_UINT32:
    case ::onnx::TensorProto_DataType_INT32:
      return 4;
    case ::onnx::TensorProto_DataType_FLOAT:
      return sizeof(float);
    case ::onnx::TensorProto_DataType_DOUBLE:
      return sizeof(double);
    case ::onnx::TensorProto_DataType_INT64:
    case ::onnx::TensorProto_DataType_UINT64:
    case ::onnx::TensorProto_DataType_COMPLEX64:
      return 8;
    case ::onnx::TensorProto_DataType_COMPLEX128:
      return 16;
    case ::onnx::TensorProto_DataType_STRING:
      return sizeof(char *);
    default:
      return 0;
  }
}

//===----------------------------------------------------------------------===//
// Interpreter
//===----------------------------------------------------------------------===//
void Interpreter::visitNode(::onnx::Node *pNode) {
  errs() << "Not implemented node: " << pNode->kind().toString() << std::endl;
}

${interpreter_visitors}
