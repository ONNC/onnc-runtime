include_directories(${PROJECT_SOURCE_DIR}/SkyPat/include)
include_directories(${PROJECT_SOURCE_DIR}/src/include)
link_directories(${PROJECT_BINARY_DIR})
link_directories(${PROJECT_BINARY_DIR}/SkyPat/install/lib)
include(ExternalProject)
ExternalProject_Add(Ext_SkyPat
    PREFIX SkyPat
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/SkyPat
    CONFIGURE_COMMAND ./autogen.sh COMMAND ./configure --prefix=${PROJECT_BINARY_DIR}/SkyPat/install
    BUILD_COMMAND make
    INSTALL_COMMAND make install
    BUILD_IN_SOURCE 1
)

add_subdirectory(unittests)