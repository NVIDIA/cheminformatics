# Copyright 2020 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
FROM rapidsai/rapidsai:0.16-cuda11.0-runtime-ubuntu18.04-py3.8

# install to rapids virtual environment
RUN conda install -c rdkit -n rapids rdkit

# install to rapids virtual environment using pip
RUN /opt/conda/envs/rapids/bin/pip install chembl_webresource_client

RUN /opt/conda/envs/rapids/bin/pip install \
    dash \
    jupyter-dash \
    dash_bootstrap_components \
    dash_core_components \
    dash_html_components \
    progressbar2 \
    tables \
    sqlalchemy && \
    pip install --ignore-installed --upgrade tensorflow==1.13.1 tensorflow-gpu==1.13.1

# misc python packages
RUN conda install -n rapids pywget

# plotly
RUN conda install -n rapids -c plotly plotly=4.9.0

# Copy source code
RUN mkdir -p /opt/nvidia/cheminfomatics/
WORKDIR /opt/nvidia/cheminfomatics/
RUN git clone git@github.com:jrwnter/cddd.git && \
    cd cddd && \
    /opt/conda/envs/rapids/bin/pip install -e . && \
    ./download_default_model.sh
    

COPY launch.sh /opt/nvidia/cheminfomatics/
COPY *.py /opt/nvidia/cheminfomatics/
COPY *.ipynb /opt/nvidia/cheminfomatics/

ENV UCX_LOG_LEVEL error

CMD /opt/nvidia/cheminfomatics/launch.sh dash
