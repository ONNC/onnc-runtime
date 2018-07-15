FROM ubuntu:16.04

RUN groupadd --gid 1000 onnc \
    && useradd --uid 1000 --gid onnc --shell /bin/bash --create-home onnc \
    && mkdir -p /etc/sudoers.d \
    && echo 'onnc ALL=(ALL:ALL) NOPASSWD:ALL' >> /etc/sudoers.d/onnc \
    && chmod 440 /etc/sudoers.d/onnc

RUN sed -i 's/archive.ubuntu.com/debian.linux.org.tw/' /etc/apt/sources.list \
    && apt-get update \
    && apt-get install -y --no-install-recommends \
        curl \
        sudo \
        make \
        g++ \
        python \
# for onnx
        git \
        protobuf-compiler \
        libprotoc-dev \
        python-pip \
        python-dev \
        python-setuptools \
# for Debug
        gdb \
    && rm -rf /var/lib/apt/lists/*

VOLUME [ "/home/onnc/onnc-runtime" ]

WORKDIR /home/onnc/onnc-runtime
RUN sudo chown onnc:onnc /home/onnc/onnc-runtime

COPY --chown=onnc:onnc ./ ./

USER onnc