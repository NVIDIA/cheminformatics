# Copyright 2020 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
FROM ubuntu:18.04
RUN apt update && DEBIAN_FRONTEND=noninteractive apt-get install -y wget git

SHELL ["/bin/bash", "-c"]
RUN  wget --quiet -O /tmp/miniconda.sh \
    https://repo.anaconda.com/miniconda/Miniconda3-py37_4.9.2-Linux-x86_64.sh \
    && /bin/bash /tmp/miniconda.sh -b -p /opt/conda \
    && rm /tmp/miniconda.sh \
    && ln -s /opt/conda/etc/profile.d/conda.sh /etc/profile.d/conda.sh
ENV PATH /opt/conda/bin:$PATH

# Copy conda env spec.
COPY setup/cuchem_rapids_0.17.yml /tmp

RUN conda env create --name cuchem -f /tmp/cuchem_rapids_0.17.yml
ENV PATH /opt/conda/envs/cuchem/bin:$PATH
RUN conda clean -afy
RUN rm /tmp/cuchem_rapids_0.17.yml

RUN source activate cuchem && python3 -m ipykernel install --user --name=cuchem
RUN echo "source activate cuchem" > /etc/bash.bashrc

COPY launch.sh /opt/nvidia/cheminfomatics/
COPY *.py /opt/nvidia/cheminfomatics/
COPY nbs/*.ipynb /opt/nvidia/cheminfomatics/

ENV UCX_LOG_LEVEL error

CMD /opt/nvidia/cheminfomatics/launch.sh dash
