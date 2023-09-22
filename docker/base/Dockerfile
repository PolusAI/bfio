FROM ubuntu:focal

# Polus platform file extension
ENV POLUS_EXT=".ome.tif"

# Container log level
ENV POLUS_LOG="INFO"

# Data and code directories
ENV EXEC_DIR="/opt/executables"
ENV DATA_DIR="/data"

RUN mkdir ${EXEC_DIR} && \
    mkdir ${DATA_DIR}} && \
    mkdir /bfio_src

# Install Python
RUN apt update && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt install python3.9 python3.9-distutils curl openjdk-8-jre -y && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py && \
    apt autoremove -y && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.9 /usr/bin/python3

# Copy the source
COPY . /bfio_src

# Install
RUN pip3 install /bfio_src[all] --no-cache-dir && \
    rm -rf /bfio_src
