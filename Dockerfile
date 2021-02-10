# Copyright 2020 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
FROM ubuntu:18.04
RUN apt update && DEBIAN_FRONTEND=noninteractive apt-get install -y wget git

SHELL ["/bin/bash", "-c"]
RUN  wget --quiet -O /tmp/miniconda.sh \
    https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh \
    && /bin/bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && /opt/conda/bin/conda clean -tipsy \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh

# Copy source code
RUN mkdir -p /opt/nvidia/cuchemsetup
COPY setup/cuchem_rapids_0.17.yml /opt/nvidia/cuchemsetup/

RUN /opt/conda/bin/conda env create --name cuchem -f /opt/nvidia/cuchemsetup/cuchem_rapids_0.17.yml

RUN cd /opt/conda/bin/ && source activate cuchem && python3 -m ipykernel install --user --name=cuchem
RUN echo "cd /opt/conda/bin/ && source activate cuchem" > ~/.bashrc

COPY launch.sh /opt/nvidia/cheminfomatics/
COPY *.py /opt/nvidia/cheminfomatics/
COPY nbs/*.ipynb /opt/nvidia/cheminfomatics/

ENV UCX_LOG_LEVEL error
ENV PATH /opt/conda/envs/cuchem/bin:$PATH

CMD /opt/nvidia/cheminfomatics/launch.sh dash
