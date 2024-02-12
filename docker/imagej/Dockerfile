# Get Linux
FROM maven:3.8.6-jdk-11-slim

# Copy the source
COPY . /bfio_src

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

# Install
RUN pip3 install /bfio_src --no-cache-dir && \
    rm -rf /bfio_src

# Install packages
ENV LIBRARY_PATH=/lib:/usr/lib
RUN pip3 install cython --no-cache-dir && \
    pip3 install pyimagej==1.4.1 --no-cache-dir

# Polus platform file extension
ENV POLUS_IMG_EXT=".ome.tif"
ENV POLUS_TAB_EXT=".csv"

# Container log level
ENV POLUS_LOG="INFO"

# Data and code directories
ENV EXEC_DIR="/opt/executables"
ENV DATA_DIR="/data"

RUN mkdir ${EXEC_DIR} && \
    mkdir ${DATA_DIR}

# create a non-root user named polusai
RUN groupadd polusai && \
    useradd --no-log-init -r -m -g polusai polusai && \
    usermod -aG sudo polusai

ENV HOME=/home/polusai
RUN chmod -R o+rx ${HOME}
# NOTE: By default, CWL adds --user=... to the docker run commands.
# https://github.com/common-workflow-language/cwltool/blob/7b8f174828e3c07b3067e4f88568124f7c878d2d/cwltool/docker.py#L362-L366
# Since we do not control the userid and/or groupid on the
# host machine(s) where this docker image will be executed,
# we must make HOME readable and executable for all users! Otherwise:
# /usr/bin/python3: Error while finding module specification for 'polus.plugins...'
# (ModuleNotFoundError: No module named 'polus')

RUN chown -R polusai ${EXEC_DIR} && \
    chown -R polusai ${DATA_DIR} && \
    chmod -R u+rwx ${EXEC_DIR} && \
    chmod -R u+rwx ${DATA_DIR}

USER polusai
RUN python3 -c 'import bfio; bfio.start()'
RUN python3 -c 'import imagej; ij = imagej.init(); print(ij.getVersion())'
