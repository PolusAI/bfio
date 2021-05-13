#!/bin/bash

version=$(<VERSION)

# Python builds
docker build . -f ./docker/slim-buster/DockerfilePython -t labshare/polus-bfio-util:${version} -t labshare/polus-bfio-util:${version}-python
# docker build . -f ./docker/tensorflow/DockerfilePython -t labshare/polus-bfio-util:${version}-tensorflow

# Java builds
docker build . -f ./docker/slim-buster/DockerfileJava -t labshare/polus-bfio-util:${version}-java
docker build . -f ./docker/imagej/Dockerfile -t labshare/polus-bfio-util:${version}-imagej
