# onnc-runtime

Runtime for onnc compiler

# Introduction

# Prerequisites

* CMake >= 3.5
* python 2.7
* gcc
* g++
* git
* [Optional] docker

### Ubuntu - with apt

````
sudo apt install make cmake g++ gcc python2.7 git protobuf-compiler libprotoc-dev python-pip python-dev python-setuptools
````

# Setup

### Update submodule [Required]

````
git submodule init
git submodule update
````

### Docker [Optional]

````
mkdir build
cmake ..
make docker-image
make docker-container
````

`docker-image` will build our docker image to your local repository as `onnc/onnc-runtime`.

`docker-container` will create a container and bind the project directory to `/home/onnc/onnc-runtime`. You can also create  container in your own way

# Build

### Legacy

````
mkdir build
cmake ..
make
````

### Docker

````
mkdir build
cmake ..
make
````

# Run