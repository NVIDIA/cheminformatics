# Copyright 2022 NVIDIA Corporation
FROM gitlab-master.nvidia.com/mlivne/nemo_containers:megamolbart_training_nemo_latest
ARG REPO_BRANCH=main

RUN apt-get update \
    && apt-get upgrade -y \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y wget git unzip tmux libxrender1 \
    && rm -rf /var/lib/apt/lists/*

## Environment setup
SHELL ["conda", "run", "-n", "base", "/bin/bash", "-c"]
COPY requirements.txt /tmp/requirements.txt
RUN pip install -r /tmp/requirements.txt
RUN pip install --upgrade numpy

RUN mkdir -p /opt/nvidia && \
    git clone https://github.com/NVIDIA/cheminformatics.git \
        --branch ${REPO_BRANCH} /opt/nvidia/cheminformatics

ENV PYTHONPATH /opt/nvidia/cheminformatics/benchmark