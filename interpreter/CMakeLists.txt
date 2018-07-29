option(BUILD_ONNCI "Build onnc-runtime interpreter" ON)
if(BUILD_ONNCI)
include_directories(${PROJECT_SOURCE_DIR}/src/include)
link_directories(${CMAKE_BINARY_DIR})

find_package(onnx REQUIRED)
#add_executable(onnci
#    main.cpp
#)
#target_link_libraries(onnci
#    onnx
#    onnx_proto
#    protobuf
#    onnc-runtime
#)
#install(TARGETS onnci
#    RUNTIME DESTINATION bin
#    LIBRARY DESTINATION lib
#    ARCHIVE DESTINATION lib/static
#    PUBLIC_HEADER DESTINATION include
#)
endif(BUILD_ONNCI)