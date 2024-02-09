FROM ubuntu:jammy

# Polus platform file extension
ENV POLUS_EXT=".ome.tif"

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
    apt install curl maven openjdk-8-jre -y && \
    curl https://bootstrap.pypa.io/get-pip.py -o get-pip.py && \
    python3 get-pip.py && \
    apt autoremove -y && \
    rm -rf /var/lib/apt/lists/*

# Copy the source
COPY . /bfio_src

# Install
RUN pip3 install /bfio_src --no-cache-dir && \
    rm -rf /bfio_src

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
