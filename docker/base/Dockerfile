FROM ubuntu:jammy

# Polus platform file extension
ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".csv"

# Container log level
ENV POLUS_LOG="INFO"

# Data and code directories
ENV EXEC_DIR="/opt/executables"
ENV DATA_DIR="/data"

# This is necessary so we don't hang on transitive dep
# or reconfiguration triggers ignoring -y
ENV DEBIAN_FRONTEND=noninteractive

ARG PYTHON_VERSION=3.11

RUN mkdir ${EXEC_DIR} && \
    mkdir ${DATA_DIR} && \
    mkdir /bfio_src


RUN apt update && \
    apt install -y software-properties-common curl git maven openjdk-8-jre && \
    add-apt-repository ppa:deadsnakes/ppa -y && \
    apt update && \
    apt install -y \
        python${PYTHON_VERSION} \
        python${PYTHON_VERSION}-venv \
        python3-distutils && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python${PYTHON_VERSION} get-pip.py && \
    rm get-pip.py && \
    apt autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Make python3 point at the new version
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python${PYTHON_VERSION} 1 && \
    update-alternatives --set python3 /usr/bin/python${PYTHON_VERSION}

# Copy the source
COPY . /bfio_src

# Apply m2 repo patch
RUN git apply --directory=bfio_src /bfio_src/docker/m2_repo.patch

# Install using the new Python
RUN python${PYTHON_VERSION} -m pip install /bfio_src --no-cache-dir

# Apply jgo copy patch
RUN git apply \
    --directory=usr/local/lib/python${PYTHON_VERSION}/dist-packages \
    /bfio_src/docker/jgo_copy.patch

# clean up
RUN rm -rf /bfio_src

# Update Maven local repository path
RUN mkdir -p /usr/share/maven/conf && \
    echo '<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0" ' \
         'xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" ' \
         'xsi:schemaLocation="http://maven.apache.org/SETTINGS/1.0.0 ' \
         'http://maven.apache.org/xsd/settings-1.0.0.xsd">' \
         '<localRepository>/etc/m2</localRepository>' \
         '</settings>' > /usr/share/maven/conf/settings.xml

# Ensure the /etc/m2 directory exists
RUN mkdir -p /etc/m2

# Set the correct permissions
RUN chmod -R 777 ${EXEC_DIR} && \
    chmod -R 777 ${DATA_DIR} && \
    chmod -R 777 /etc/m2

RUN python3 -c 'import bfio; bfio.start()'

# Set the correct permissions again, for sanity
RUN chmod -R 777 /etc/m2
