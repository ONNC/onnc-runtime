include_directories(${PROJECT_SOURCE_DIR}/src/include)
link_directories(${CMAKE_BINARY_DIR})

add_executable(onnci
    main.cpp
)
target_link_libraries(onnci
    onnx
    onnx_proto
    protobuf
    onnc-runtime
)

add_dependencies(onnci
    ext_project_onnx
)

install(TARGETS onnci
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static
    PUBLIC_HEADER DESTINATION include
)