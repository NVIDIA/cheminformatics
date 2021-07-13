ARG BASE_IMAGE=nvcr.io/nvidia/tritonserver:21.03-py3
FROM $BASE_IMAGE

RUN apt update && DEBIAN_FRONTEND=noninteractive apt-get install -y wget git unzip python3-rdkit\
    && rm -rf /var/lib/apt/lists/*

# TODO: Replace the following lines once the repo is available.
RUN wget  --quiet -O /tmp/pysmilesutils-mirror.tgz \
    http://rilango-work.nvidia.com/molbart/pysmilesutils-mirror.tgz \
    && tar -xf /tmp/pysmilesutils-mirror.tgz -C /opt \
    && cd /opt/pysmilesutils-mirror; pip install .

# TODO: Remove once MegaMolBART is complete ?
RUN wget  --quiet -O /tmp/molbart-mirror.tgz \
    http://rilango-work.nvidia.com/molbart/molbart-mirror.tgz \
    && tar -xf /tmp/molbart-mirror.tgz -C /opt \
    && cd /opt/molbart-mirror; pip install -r requirements.txt; pip install .

# TODO: Remove once MegaMolBART is complete ?
RUN mkdir -p /models/molbart \
    && wget --quiet -O /models/molbart/mol_opt_tokeniser.pickle \
    http://rilango-work.nvidia.com/molbart/mol_opt_tokeniser.pickle \
    && wget --quiet -O /models/molbart/az_molbart_pretrain.ckpt \
    http://rilango-work.nvidia.com/molbart/az_molbart_pretrain.ckpt

WORKDIR /workspace/
COPY triton/molbart/model.py /workspace/models/molbart/1/
COPY triton/molbart/config.pbtxt /workspace/models/molbart/
COPY triton/molbart/client.py /workspace/models/molbart/

CMD tritonserver --model-repository /workspace/models