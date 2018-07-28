execute_process (COMMAND python -c "from distutils.sysconfig import get_python_lib; print get_python_lib()" 
    OUTPUT_VARIABLE PYTHON_SITE_PACKAGES 
    OUTPUT_STRIP_TRAILING_WHITESPACE
)
message(STATUS ${PYTHON_SITE_PACKAGES})

find_library(ONNX_LIBRARIES
	NAMES onnx
    HINTS ${ONNC_RUNTIME_ROOT}/onnx/build
    HINTS ${PYTHON_SITE_PACKAGES}/onnx
)
message(STATUS ${ONNX_LIBRARIES})
find_path(ONNX_INCLUDE_DIR
	NAMES onnx.pb.h
	HINTS ${ONNC_RUNTIME_ROOT}/onnx/build/onnx
)
message(STATUS ${ONNX_INCLUDE_DIR})