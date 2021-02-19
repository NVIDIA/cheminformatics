# Copyright 2020 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
FROM nvidia/cuda:11.0-base
ARG CODE_BASE=/opt/nvidia
ARG PACKAGE_PATH=${CODE_BASE}/cheminformatics/
ARG CONDA_ENV=cuchem

RUN apt update && \
    DEBIAN_FRONTEND=noninteractive \
    && apt-get install -y wget git unzip \
    && rm -rf /var/lib/apt/lists/*

# Install conda
SHELL ["/bin/bash", "-c"]
RUN wget --quiet -O /tmp/miniconda.sh \
    https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh \
    && /bin/bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
ENV PATH /opt/conda/bin:$PATH

# Setup conda env
COPY setup/cuchem_rapids_0.17.yml /tmp

RUN conda env create --name ${CONDA_ENV} -f /tmp/cuchem_rapids_0.17.yml \
    && rm /tmp/cuchem_rapids_0.17.yml \
    && conda clean -ay
ENV PATH /opt/conda/envs/${CONDA_ENV}/bin:$PATH

RUN conda run -n ${CONDA_ENV} python3 -m ipykernel install --user --name=${CONDA_ENV}
RUN echo "source activate cuchem" > /etc/bash.bashrc

# Add cddd code
COPY launch.sh ${PACKAGE_PATH}/
COPY *.py ${PACKAGE_PATH}/
COPY nbs/*.ipynb ${PACKAGE_PATH}/
RUN /opt/conda/envs/${CONDA_ENV}/bin/pip uninstall -y cddd
RUN cd /workspace \
    && git clone https://github.com/jrwnter/cddd.git \
    && cd cddd \
    && /opt/conda/envs/${CONDA_ENV}/bin/pip install -e . \ 
    && ./download_default_model.sh

ENV UCX_LOG_LEVEL error

CMD ${PACKAGE_PATH}/launch.sh dash
