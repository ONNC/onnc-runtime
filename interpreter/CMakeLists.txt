option(BUILD_ONNCI "Build onnc-runtime interpreter" ON)

if(BUILD_ONNCI)
include_directories(${ONNC_RUNTIME_ROOT}/src/include)
link_directories(${CMAKE_BINARY_DIR})
find_package(onnx REQUIRED)

endif(BUILD_ONNCI)