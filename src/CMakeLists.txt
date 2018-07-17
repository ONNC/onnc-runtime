include_directories(include)
add_library(onnc-runtime SHARED
    lib/onnc-runtime.c
    lib/onnc-runtime-operator.c
)
set_target_properties(onnc-runtime
    PROPERTIES OUTPUT_NAME onnc-runtime
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}
)

install(TARGETS onnc-runtime
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static
    PUBLIC_HEADER DESTINATION include
)