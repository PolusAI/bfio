# Get Linux
FROM tensorflow/tensorflow:2.13.0-gpu

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

# Install Java and Maven
RUN apt update && \
    apt install software-properties-common -y && \
    apt install git maven openjdk-8-jre -y && \
    apt autoremove -y && \
    rm -rf /var/lib/apt/lists/*

RUN pip3 install --upgrade pip --no-cache-dir

# Copy the source
COPY . /bfio_src

# Apply m2 repo patch
RUN git apply --directory=bfio_src /bfio_src/docker/m2_repo.patch

# Install
RUN pip3 install /bfio_src --no-cache-dir && \
    rm -rf /bfio_src

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

# Set the correct permissions again, for sanity
RUN chmod -R 777 /etc/m2