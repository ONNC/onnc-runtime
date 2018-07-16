cmake_minimum_required(VERSION 3.5)
project(onnc-runtime)

set(C_STANDARD 11)
set(CXX_STANDARD 14)
add_compile_options(-fPIC)
add_compile_options(-Ofast)
add_compile_options(-ffast-math)

add_subdirectory(docker)
add_subdirectory(src)

include_directories(src/include)

add_executable(onnc-runtime-bin
    main.cpp
)

target_link_libraries(onnc-runtime-bin
    onnc-runtime
)
set_target_properties(onnc-runtime-bin
    PROPERTIES OUTPUT_NAME onnc-runtime
)

option(CMAKE_INSTALL_PREFIX "Install prefix" ${PROJECT_SOURCE_DIR}/install)

install(TARGETS onnc-runtime-bin
    RUNTIME DESTINATION bin
    LIBRARY DESTINATION lib
    ARCHIVE DESTINATION lib/static
    PUBLIC_HEADER DESTINATION include
)