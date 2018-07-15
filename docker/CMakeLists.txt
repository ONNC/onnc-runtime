add_custom_target(docker-image
    COMMAND docker build -t onnc/onnc-runtime ${CMAKE_CURRENT_LIST_DIR}
)

add_custom_target(docker-container
    COMMAND docker create "--name" onnc-runtime ${PROJECT_SOURCE_DIR}:/home/onnc/onnc-runtime onnc/onnc-runtime
)