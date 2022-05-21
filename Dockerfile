# Copyright 2022 NVIDIA Corporation
FROM rapidsai/rapidsai-core-dev:22.04-cuda11.5-devel-ubuntu20.04-py3.8
ARG REPO_BRANCH=main

RUN apt-get update \
    && apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y wget git unzip tmux libxrender1 \
    && rm -rf /var/lib/apt/lists/*

## Environment setup
SHELL ["conda", "run", "-n", "rapids", "/bin/bash", "-c"]
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt

RUN mkdir -p /opt/nvidia && \
    git clone https://github.com/NVIDIA/cheminformatics.git \
        --branch ${REPO_BRANCH} /opt/nvidia/cheminformatics

ENV PYTHONPATH /opt/nvidia/cheminformatics/benchmark