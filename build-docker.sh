#!/bin/bash

version=$(<VERSION)

# Python builds
docker build . -f ./docker/base/Dockerfile -t polusai/bfio:"${version}"
docker build . -f ./docker/tensorflow/Dockerfile -t polusai/bfio:"${version}"-tensorflow
docker build . -f ./docker/imagej/Dockerfile -t polusai/bfio:"${version}"-imagej
