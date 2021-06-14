#!/bin/bash

version=$(<VERSION)

# Python builds
docker build . -f ./docker/base/Dockerfile -t labshare/polus-bfio-util:${version}
docker build . -f ./docker/tensorflow/Dockerfile -t labshare/polus-bfio-util:${version}-tensorflow
docker build . -f ./docker/imagej/Dockerfile -t labshare/polus-bfio-util:${version}-imagej
