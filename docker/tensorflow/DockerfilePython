# Get Linux
FROM tensorflow/tensorflow:2.4.1-gpu

RUN pip3 install --upgrade pip --no-cache-dir

RUN python --version

### Things that change rarely between containers

# Create temporary folder
ARG DATA_DIR="/data"

# Copy the requirements file
COPY ./requirements/requirements-base.txt ${EXEC_DIR}/

# Install bfio and requirements
RUN pip3 install tifffile zarr && \
    pip3 install -r ${EXEC_DIR}/requirements-base.txt --no-cache-dir --no-dependencies && \
    rm -rf /usr/local/lib/python3.9/site-packages/bfio/jars