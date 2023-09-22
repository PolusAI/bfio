# Get Linux
FROM tensorflow/tensorflow:2.13.0-gpu

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
RUN pip3 install /bfio_src[all] --no-cache-dir --no-dependencies && \
    rm -rf /bfio_src
