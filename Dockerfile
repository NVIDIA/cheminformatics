# Copyright 2020 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
FROM nvidia/cuda:11.0-base

RUN apt update \
    && DEBIAN_FRONTEND=noninteractive apt-get install -y wget git unzip \
    && apt install --only-upgrade libssl1.1 openssl \
    && rm -rf /var/lib/apt/lists/*

SHELL ["/bin/bash", "-c"]
RUN  wget --quiet -O /tmp/miniconda.sh \
    https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh \
    && /bin/bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

ENV PATH /opt/conda/bin:$PATH

# Copy conda env spec.
COPY setup/cuchem_rapids_0.17.yml /tmp

RUN conda env create --name cuchem -f /tmp/cuchem_rapids_0.17.yml \
    && rm /tmp/cuchem_rapids_0.17.yml\
    && conda clean -afy
ENV PATH /opt/conda/envs/cuchem/bin:$PATH

RUN source activate cuchem && python3 -m ipykernel install --user --name=cuchem
RUN echo "source activate cuchem" > /etc/bash.bashrc

ADD "https://www.random.org/cgi-bin/randbyte?nbytes=10&format=h" skipcache
RUN mkdir -p /opt/nvidia/ \
    && cd /opt/nvidia/ \
    && git clone --single-branch --branch master https://github.com/NVIDIA/cheminformatics.git cheminfomatics \
    && rm -rf /opt/nvidia/cheminfomatics/.git

ENV UCX_LOG_LEVEL error
EXPOSE 5000

CMD cd /opt/nvidia/cheminfomatics; /opt/nvidia/cheminfomatics/launch.sh dash