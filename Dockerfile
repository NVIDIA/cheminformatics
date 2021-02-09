# Copyright 2020 NVIDIA Corporation
# SPDX-License-Identifier: Apache-2.0
FROM rapidsai/rapidsai-dev:0.17-cuda10.1-devel-ubuntu18.04-py3.7

# install to rapids virtual environment
RUN conda install -n rapids -c rdkit -c plotly plotly=4.9.0 pywget rdkit

# install to rapids virtual environment using pip
RUN /opt/conda/envs/rapids/bin/pip install \
    dash \
    jupyter-dash \
    dash_bootstrap_components \
    dash_core_components \
    dash_html_components \
    progressbar2 \
    tables \
    sqlalchemy \
    openpyxl \
    tabulate \
    autopep8 \
    chembl_webresource_client

RUN /opt/conda/envs/rapids/bin/pip install --ignore-installed --upgrade \
        tensorflow-gpu==1.15.4

# Copy source code
RUN mkdir -p /opt/nvidia/cheminfomatics/
WORKDIR /opt/nvidia/cheminfomatics/
RUN git clone https://github.com/jrwnter/cddd.git && \
    cd cddd && \
    /opt/conda/envs/rapids/bin/pip install -e . && \
    ./download_default_model.sh

COPY launch.sh /opt/nvidia/cheminfomatics/
COPY *.py /opt/nvidia/cheminfomatics/
COPY *.ipynb /opt/nvidia/cheminfomatics/

ENV UCX_LOG_LEVEL error

CMD /opt/nvidia/cheminfomatics/launch.sh dash
