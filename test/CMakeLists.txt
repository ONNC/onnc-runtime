option(BUILD_ONNC_RUNTIME_TESTS "Build onnc-runtime tests" ON)
if(BUILD_ONNC_RUNTIME_TESTS)

find_package(SkyPat REQUIRED)
include_directories(${ONNC_RUNTIME_ROOT}/src/include)
add_subdirectory(unittests)

endif(BUILD_ONNC_RUNTIME_TESTS)