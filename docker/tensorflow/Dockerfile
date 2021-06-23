# Get Linux
FROM tensorflow/tensorflow:2.4.1-gpu

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

# Copy the requirements file
COPY ./requirements/requirements-base.txt ${EXEC_DIR}/

# Install bfio and requirements
RUN pip3 install -r ${EXEC_DIR}/requirements-base.txt --no-cache-dir --no-dependencies && \
    rm -rf /usr/local/lib/python3.9/site-packages/bfio/jars
