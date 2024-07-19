# Get Linux
FROM maven:3.8.6-jdk-11-slim

# Polus platform file extension
ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".csv"

# Container log level
ENV POLUS_LOG="INFO"

# Data and code directories
ENV EXEC_DIR="/opt/executables"
ENV DATA_DIR="/data"

RUN mkdir ${EXEC_DIR} && \
    mkdir ${DATA_DIR} && \
    mkdir /bfio_src


# Install Python
RUN apt update && \
    apt install software-properties-common -y && \
    add-apt-repository ppa:deadsnakes/ppa && \
    apt install python3.9 python3.9-distutils curl git -y && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3.9 get-pip.py && \
    apt autoremove -y && \
    rm -rf /var/lib/apt/lists/*

RUN ln -sf /usr/bin/python3.9 /usr/bin/python3


# Copy the source
COPY . /bfio_src

# Apply m2 repo patch
RUN git apply --directory=bfio_src /bfio_src/docker/m2_repo.patch

# Install
RUN pip3 install /bfio_src --no-cache-dir

#Apply jgo copy patch
RUN  git apply  --directory=usr/local/lib/python3.9/dist-packages /bfio_src/docker/jgo_copy.patch

# Install packages
ENV LIBRARY_PATH=/lib:/usr/lib
RUN pip3 install cython --no-cache-dir && \
    pip3 install pyimagej==1.4.1 --no-cache-dir

#Apply pyimagej patch
RUN  git apply  --directory=usr/local/lib/python3.9/dist-packages /bfio_src/docker/pyimagej_m2repo.patch

# clean up
RUN rm -rf /bfio_src

# Update Maven local repository path
RUN mkdir -p /usr/share/maven/conf
RUN echo '<settings xmlns="http://maven.apache.org/SETTINGS/1.0.0" ' \
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
RUN python3 -c 'import imagej; ij = imagej.init(); print(ij.getVersion())'

# Set the correct permissions again, for sanity
RUN chmod -R 777 /etc/m2