#include <iostream>
#include <fstream>
#include <cstddef>
#include <cstdio>
#include <cstring>
#include <map>
#include <onnx/onnx.pb.h>
#include <onnx/shape_inference/implementation.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

extern "C"{
#include <onnc-runtime-internal.h>
}

typedef std::map<std::string, void*> address_table_t;
typedef std::map<std::string, const ::onnx::TensorShapeProto *> shape_table_t;

void prepareWeightFile(const char *fileName, const ::onnx::GraphProto& graph){
    struct {
        unsigned long long len;
    } initializer_len;
    struct {
        unsigned long long offset;
        unsigned long long size;
    } offset_struct;

    std::ofstream weight_fout(fileName, std::ios_base::out | std::ios_base::binary);
    const char *magic = ".TSR\0\0\0\0";

    weight_fout.write(magic, 8);
    size_t offset = 16 * (1 + graph.initializer_size());
    initializer_len.len = graph.initializer_size();
    weight_fout.write((const char *)&initializer_len, sizeof(initializer_len));
    // Prepare offset table
    for(int i = 0; i < graph.initializer_size(); ++i){
        const ::onnx::TensorProto& tensor = graph.initializer(i);
        offset_struct.offset = offset;
        offset_struct.size = tensor.raw_data().size();
        weight_fout.write((const char *)&offset_struct, sizeof(offset_struct));
        offset += offset_struct.size;
    }
    // Write data
    for(int i = 0; i < graph.initializer_size(); ++i){
        const std::string& rawData = graph.initializer(i).raw_data();
        weight_fout.write(rawData.data(), rawData.size());
    }
    weight_fout.close();
}

size_t sizeof_tensor_type(const ::onnx::TensorProto_DataType& dataType){
    switch(dataType){
        case ::onnx::TensorProto_DataType_BOOL:
            return sizeof(bool);
        case ::onnx::TensorProto_DataType_INT8:
        case ::onnx::TensorProto_DataType_UINT8:
        case ::onnx::TensorProto_DataType_STRING:
            return sizeof(int8_t);
        case ::onnx::TensorProto_DataType_UINT16:
        case ::onnx::TensorProto_DataType_INT16:
        case ::onnx::TensorProto_DataType_FLOAT16:
            return sizeof(int16_t);
        case ::onnx::TensorProto_DataType_UINT32:
        case ::onnx::TensorProto_DataType_INT32:
            return sizeof(int32_t);
        case ::onnx::TensorProto_DataType_FLOAT:
            return sizeof(float);
        case ::onnx::TensorProto_DataType_DOUBLE:
            return sizeof(double);
        case ::onnx::TensorProto_DataType_INT64:
        case ::onnx::TensorProto_DataType_UINT64:
        case ::onnx::TensorProto_DataType_COMPLEX64:
            return sizeof(int64_t);
        case ::onnx::TensorProto_DataType_COMPLEX128:
            return sizeof(int64_t) * 2;
        default:
            return 0;
    }
}

void allocate_output_memory(Context const *context, const ::onnx::ValueInfoProto& valueInfo, address_table_t& address_table, shape_table_t& shape_table){
    const ::onnx::TypeProto_Tensor& tensor_type = valueInfo.type().tensor_type();
    size_t size = 1;
    for(int i = 0; i < tensor_type.shape().dim_size(); ++i){
        const ::onnx::TensorShapeProto& shape = tensor_type.shape();
        size *= shape.dim(i).dim_value();
    }
    address_table[valueInfo.name()] = ONNC_RUNTIME_internal_allocate_memory((void *)context, size, sizeof_tensor_type(tensor_type.elem_type()));
}

int main(int argc, char *argv[]){
    // Check args
    if(argc < 3){
        std::cerr << "Usage: onnci <ONNX module file> <input file>" << std::endl;
        return -1;
    }
    // Read onnx module
    ::onnx::ModelProto model;
    std::ifstream model_fin(argv[1]);
    ::google::protobuf::io::IstreamInputStream model_input_stream(&model_fin);
    ::google::protobuf::io::CodedInputStream model_coded_stream(&model_input_stream);
    model_coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
    model.ParseFromCodedStream(&model_coded_stream);
    ::onnx::shape_inference::InferShapes(model);
    const ::onnx::GraphProto& graph = model.graph();

    // Prepare weight
    const char *weight_filename = "weight.onnc.bin";
    prepareWeightFile(weight_filename, graph);

    // Init runtime
    Context *context = (Context *)ONNC_RUNTIME_init_runtime(weight_filename);

    address_table_t address_table;
    shape_table_t shapes_table;

    // Load weight
    for(int i = 0; i < graph.initializer_size(); ++i){
        address_table[graph.initializer(i).name()] = ONNC_RUNTIME_load_weight(context, i);
    }
    
    // Load input
    for(int i = 0; i < graph.input_size(); ++i){
        const ::onnx::ValueInfoProto& valueInfo = graph.input(i);
        shapes_table[valueInfo.name()] = &(valueInfo.type().tensor_type().shape());
        if(address_table.find(valueInfo.name()) == address_table.end()){
            ::onnx::TensorProto tensor;
            std::ifstream input_fin(argv[2]);
            tensor.ParseFromIstream(&input_fin);
            const std::string& raw_data_str = tensor.raw_data();
            int8_t *raw_data = new int8_t[raw_data_str.size()];
            address_table[valueInfo.name()] = memcpy(raw_data, raw_data_str.data(), sizeof(int8_t) * raw_data_str.size());
        }
    }

    // Allocate output memory
    for(int i = 0; i < graph.value_info_size(); ++i){
        allocate_output_memory(context, graph.value_info(i), address_table, shapes_table);
    }
    for(int i = 0; i < graph.output_size(); ++i){
        allocate_output_memory(context, graph.output(i), address_table, shapes_table);
    }

    // Clean
    int ret = ONNC_RUNTIME_shutdown_runtime(context);
    remove(weight_filename);
    return ret;
}