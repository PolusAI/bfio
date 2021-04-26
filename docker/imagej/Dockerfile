# Get Linux
FROM maven:3-jdk-8-slim

### Things that change rarely between containers

# Copy Python 3.9
COPY --from=python:3.9 / /

# Copy the requirements file
COPY ./requirements/requirements-base.txt /

# Install requirements
RUN apt-get update && \
    pip3 install -r  /requirements-base.txt --no-cache-dir

# Install packages
ENV LIBRARY_PATH=/lib:/usr/lib
RUN pip3 install cython --no-cache-dir && \
    pip3 install pyimagej==1.0.0 --no-cache-dir

# RUN python3 -c 'import imagej; ij = imagej.init("sc.fiji:fiji:2.1.1+net.imagej:imagej-legacy:0.37.4"); print(ij.getVersion())'

# RUN mkdir ~/jgo && mv ~/.jgo/net.imglib2 /

FROM maven:3-jdk-8-slim

COPY --from=python:3.9-slim / /

COPY --from=0 /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages

RUN python3 -c 'import imagej; ij = imagej.init("sc.fiji:fiji:2.1.1+net.imagej:imagej-legacy:0.37.4"); print(ij.getVersion())'

RUN rm -rf ~/.m2