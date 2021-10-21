#
# Copyright (c) 2019-2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.
#

ARG FROM_IMAGE=gpuci/miniconda-cuda
ARG CUDA_VER=11.2
ARG LINUX_VERSION=ubuntu20.04
ARG IMAGE_TYPE=devel
FROM ${FROM_IMAGE}:${CUDA_VER}-${IMAGE_TYPE}-${LINUX_VERSION}

# Add a condarc for channels and override settings
RUN echo -e "\
ssl_verify: False \n\
channels: \n\
  - gpuci \n\
  - conda-forge \n\
  - nvidia \n\
  - defaults \n" > /conda/.condarc \
      && cat /conda/.condarc ;

# Create parabricks conda env and make default
RUN source activate base
RUN conda create -n rapids-21.08 -c rapidsai -c nvidia -c conda-forge \
    rapids-blazing=21.10 python=3.7 cudatoolkit=11.2
RUN conda install -y -c conda-forge -n rapids rdkit==2020.09.1.0
SHELL ["conda", "run", "-n", "rapids", "/bin/bash", "-c"]

# ADD source dest
# Create symlink for old scripts expecting `gdf` conda env
RUN ln -s /opt/conda/envs/rapids-21.08 /opt/conda/envs/gdf

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]