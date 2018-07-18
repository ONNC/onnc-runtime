include_directories(include)

file(GLOB_RECURSE SRC_C_FILES lib/*.c)
add_library(onnc-runtime SHARED
    ${SRC_C_FILES}
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