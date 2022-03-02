# Get Linux
FROM maven:3-jdk-8-slim

### Things that change rarely between containers

# Copy Python 3.9
COPY --from=python:3.9 / /

# Copy the source
COPY . /bfio_src

# Install
RUN pip3 install /bfio_src[all] --no-cache-dir && \
    rm -rf /bfio_src

# Install packages
ENV LIBRARY_PATH=/lib:/usr/lib
RUN pip3 install cython --no-cache-dir && \
    pip3 install pyimagej==1.0.0 --no-cache-dir

FROM maven:3-jdk-8-slim

COPY --from=python:3.9-slim / /

COPY --from=0 /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

# Polus platform file extension
ENV POLUS_EXT=".ome.tif"

# Container log level
ENV POLUS_LOG="INFO"

# Data and code directories
ENV EXEC_DIR="/opt/executables"
ENV DATA_DIR="/data"

RUN mkdir ${EXEC_DIR} && \
    mkdir ${DATA_DIR}}

RUN python3 -c 'import imagej; ij = imagej.init("sc.fiji:fiji:2.1.1+net.imagej:imagej-legacy:0.37.4"); print(ij.getVersion())'

RUN rm -rf ~/.m2
