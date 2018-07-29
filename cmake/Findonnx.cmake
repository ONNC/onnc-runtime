option(ONNX_PATH "Path of onnx library")
find_library(ONNX_LIBRARIES
    NAMES onnx
    HINTS ${ONNX_PATH}
    HINTS ${ONNC_RUNTIME_ROOT}/onnx/build/
)
link_directories(${ONNX_LIBRARIES})
find_path(ONNX_INCLUDE_DIR
    NAMES onnx_pb.h
    HINTS ${ONNX_PATH}
    HINTS ${ONNC_RUNTIME_ROOT}/onnx/onnx/
)
include_directories(${ONNX_INCLUDE_DIR})
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ONNX DEFAULT_MSG ONNX_INCLUDE_DIR ONNX_LIBRARIES)
