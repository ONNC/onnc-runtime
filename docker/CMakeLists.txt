add_custom_target(docker-image
    COMMAND docker build -t onnc/onnc-runtime ${CMAKE_CURRENT_LIST_DIR}
)

add_custom_target(docker-container
    COMMAND docker create --privileged=true -t "--name" onnc-runtime --volume ${PROJECT_SOURCE_DIR}:/home/onnc/onnc-runtime onnc/onnc-runtime
    COMMAND docker start onnc-runtime
    COMMAND docker exec -t -w /home/onnc/onnc-runtime onnc-runtime pip install -e onnx
    COMMAND docker stop onnc-runtime
)

add_custom_target(docker-build
    COMMAND docker start onnc-runtime
    COMMAND docker exec -t -w /home/onnc/onnc-runtime/docker onnc-runtime mkdir -p build
    COMMAND docker exec -t -w /home/onnc/onnc-runtime/docker/build onnc-runtime cmake -DCMAKE_INSTALL_PREFIX=/home/onnc/onnc-runtime/docker/install ../..
    COMMAND docker exec -t -w /home/onnc/onnc-runtime/docker/build onnc-runtime make -j4
    COMMAND docker exec -t -w /home/onnc/onnc-runtime/docker/build onnc-runtime make install
    COMMAND docker stop onnc-runtime
)

add_custom_target(docker-clean
    COMMAND docker start onnc-runtime
    COMMAND docker exec -t -w /home/onnc/onnc-runtime/docker onnc-runtime rm -rf ./build
    COMMAND docker exec -t -w /home/onnc/onnc-runtime/docker onnc-runtime rm -rf ./install
    COMMAND docker stop onnc-runtime
)