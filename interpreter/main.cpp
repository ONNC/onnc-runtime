#include <iostream>
#include <fstream>
#include <cstddef>
#include <cstdio>
#include <onnx/onnx.pb.h>
#include <onnx/shape_inference/implementation.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/io/zero_copy_stream_impl.h>

using namespace onnx;

void prepareWeightFile(const char *fileName){
    std::ofstream weight_fout(fileName, std::ios_base::out | std::ios_base::binary);
    const char *magic = ".TSR\0\0\0\0";
    weight_fout.write(magic, sizeof(int8_t) * 8);

    weight_fout.close();
}

int main(int argc, char *argv[]){
    // Check args
    if(argc < 3){
        std::cerr << "Usage: onnci <ONNX module file> <input file>" << std::endl;
        return -1;
    }
    // Read onnx module
    ModelProto *model = new ModelProto();
    std::ifstream model_fin(argv[1]);
    ::google::protobuf::io::IstreamInputStream model_input_stream(&model_fin);
    ::google::protobuf::io::CodedInputStream model_coded_stream(&model_input_stream);
    model_coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
    model->ParseFromCodedStream(&model_coded_stream);
    shape_inference::InferShapes(*model);

    // Prepare weight
    prepareWeightFile("weight.onnc.bin");

    // Clean
    remove("weight.onnc.bin");
    delete model;
    return 0;
}