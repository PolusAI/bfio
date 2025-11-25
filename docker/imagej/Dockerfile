# Get Linux with Python 3.11
# this is more lightweight than maven:3.9.x-openjdk 
FROM python:3.11-slim

# Polus platform file extension
ENV DEBIAN_FRONTEND=noninteractive
ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".csv"
# Container log level
ENV POLUS_LOG="INFO"

# Data and code directories
ENV EXEC_DIR="/opt/executables"
ENV DATA_DIR="/data"

# global variable for just for paths
ARG PYTHON_VERSION=3.11

RUN mkdir ${EXEC_DIR} && \
    mkdir ${DATA_DIR} && \
    mkdir /bfio_src

# --------------------------------------------------
# Install system deps: Java 11, Maven, git
# --------------------------------------------------
RUN apt update && \
    apt install -y \
        default-jdk \
        maven \
        git && \
    apt autoremove -y && \
    rm -rf /var/lib/apt/lists/*



RUN python3 -m pip install --upgrade pip --no-cache-dir

# Copy the source
COPY . /bfio_src

# Apply m2 repo patch
RUN git apply --directory=bfio_src /bfio_src/docker/m2_repo.patch

# Install
RUN python3 -m pip install /bfio_src --no-cache-dir

#Apply jgo copy patch
RUN git apply \
    --directory=usr/local/lib/python${PYTHON_VERSION}/site-packages \
    /bfio_src/docker/jgo_copy.patch

# Install packages
ENV LIBRARY_PATH=/lib:/usr/lib

RUN pip3 install cython --no-cache-dir && \
    pip3 install pyimagej==1.7.0 --no-cache-dir

#Apply pyimagej patch
RUN git apply \
    --directory=usr/local/lib/python${PYTHON_VERSION}/site-packages \
    /bfio_src/docker/pyimagej_m2repo.patch

# clean up
RUN rm -rf /bfio_src

# Update Maven local repository path
RUN mkdir -p /usr/share/maven/conf && \
    echo '<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0" \
          xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" \
          xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.0.0 \
          http://maven.apache.org/xsd/settings-1.0.0.xsd"> \
          <localRepository>/etc/m2</localRepository> \
          </settings>' \
          > /usr/share/maven/conf/settings.xml

# Ensure the /etc/m2 directory exists
RUN mkdir -p /etc/m2

# Set the correct permissions
RUN chmod -R 777 ${EXEC_DIR} && \
    chmod -R 777 ${DATA_DIR} && \
    chmod -R 777 /etc/m2

RUN python3 -c 'import bfio; bfio.start()'
RUN python3 -c 'import imagej; ij = imagej.init(); print(ij.getVersion())'

# Set the correct permissions again, for sanity
RUN chmod -R 777 /etc/m2
