# onnc-runtime

Runtime for onnc compiler

![CI Status](https://api.travis-ci.org/ONNC/onnc.svg?branch=master)

# Introduction

# Prerequisites

* CMake >= 3.5
* python 2.7
* gcc
* g++
* git
* automake
* protobuf
* libtool
* [Optional] docker

### Ubuntu - with apt

````
sudo apt install make cmake g++ gcc python2.7  git protobuf-compiler libprotoc-dev python-pip python-dev python-setuptools
````

# Setup

### Update submodule [Required]

````
git submodule init
git submodule update
````

# Build

### Unix/Linux/Mac

````
mkdir build
cmake ..
make
````

### Windows (MinGW)

````
mkdir build
cd build
cmake .. -G "MinGW Makefiles"
mingw32-make
````

# Run
