# Get Linux
FROM tensorflow/tensorflow:2.13.0-gpu

# Install Java and Maven
RUN apt update && \
    apt install software-properties-common -y && \
    apt install maven openjdk-8-jre -y && \
    apt autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Polus platform file extension
ENV POLUS_EXT=".ome.tif"

# Container log level
ENV POLUS_LOG="INFO"

# Data and code directories
ENV EXEC_DIR="/opt/executables"
ENV DATA_DIR="/data"

RUN mkdir ${EXEC_DIR} && \
    mkdir ${DATA_DIR}}

RUN pip3 install --upgrade pip --no-cache-dir

# Copy the source
COPY . /bfio_src

# Install
RUN pip3 install /bfio_src --no-cache-dir && \
    rm -rf /bfio_src

RUN python3 -c 'import bfio; bfio.start()'
