add_executable(onnci
    main.cpp
)

target_link_libraries(onnci
    onnx
    onnx_proto
    protobuf
)

add_dependencies(onnci
    ext_project_onnx
)