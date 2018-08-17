cmake_minimum_required(VERSION 3.5)
project(onnc-runtime)

enable_testing()

set(CMAKE_C_STANDARD 11)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS} -DONNX_NAMESPACE=onnx)
set(ONNC_RUNTIME_ROOT ${CMAKE_CURRENT_LIST_DIR})
set(CMAKE_MODULE_PATH ${ONNC_RUNTIME_ROOT}/cmake)

option(CMAKE_INSTALL_PREFIX "Install prefix" ${PROJECT_SOURCE_DIR}/install)

add_compile_options(-fPIC)
add_compile_options(-Ofast)
add_compile_options(-ffast-math)

add_subdirectory(src)
add_subdirectory(test)