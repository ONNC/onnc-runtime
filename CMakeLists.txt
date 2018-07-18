cmake_minimum_required(VERSION 3.5)
project(onnc-runtime)

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -DONNX_NAMESPACE=onnx)

option(CMAKE_INSTALL_PREFIX "Install prefix" ${PROJECT_SOURCE_DIR}/install)

add_compile_options(-fPIC)
add_compile_options(-Ofast)
add_compile_options(-ffast-math)

include(ExternalProject)
ExternalProject_Add(ext_project_onnx
    PREFIX onnx
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/onnx
    STEP_TARGETS build
    CMAKE_ARGS -DBUILD_PYTHON=OFF
    INSTALL_COMMAND mkdir -p ../../lib ../../include COMMAND cp libonnx.a libonnx_proto.a ../../lib COMMAND cp -r onnx ../../include
)
include_directories(${PROJECT_SOURCE_DIR}/onnx)
include_directories(${CMAKE_BINARY_DIR}/onnx/include)
link_directories(${CMAKE_BINARY_DIR}/onnx/lib)

add_subdirectory(src)
add_subdirectory(test)
add_subdirectory(interpreter)
add_subdirectory(docker)