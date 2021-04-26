#!/bin/bash

version=$(<VERSION)

# Python builds
# docker build . -f ./docker/alpine/DockerfilePython -t labshare/polus-bfio-util:${version} -t labshare/polus-bfio-util:${version}-alpine -t labshare/polus-bfio-util:${version}-python -t labshare/polus-bfio-util:${version}-alpine-python
docker build . -f ./docker/slim-buster/DockerfilePython -t labshare/polus-bfio-util:${version}-slim-buster -t labshare/polus-bfio-util:${version}-slim-buster-python
docker build . -f ./docker/tensorflow/DockerfilePython -t labshare/polus-bfio-util:${version}-tensorflow

# Java builds
# docker build . -f ./docker/alpine/DockerfileJava -t labshare/polus-bfio-util:${version}-java -t labshare/polus-bfio-util:${version}-alpine-java
docker build . -f ./docker/slim-buster/DockerfileJava -t labshare/polus-bfio-util:${version}-slim-buster-java
docker build . -f ./docker/imagej/Dockerfile -t labshare/polus-bfio-util:${version}-imagej
