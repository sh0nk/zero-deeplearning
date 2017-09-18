FROM ubuntu:16.04
MAINTAINER Takefumi Ota

RUN apt-get update -y && \
    apt-get upgrade -y && \
    apt-get install -yq vim curl python3 python3-pip

ADD src /srv/src
WORKDIR /srv/src
ENV PYTHONPATH /srv/src

RUN pip3 install -r requirements.txt && \
    mkdir -p /srv/data/mnist && \
    python3 /srv/src/lib/mnist.py
