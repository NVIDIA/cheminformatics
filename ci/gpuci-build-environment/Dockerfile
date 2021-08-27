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
ARG CUDA_VER=10.1
ARG LINUX_VERSION=ubuntu18.04
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

COPY ci/env.yml /tmp

# Create parabricks conda env and make default
RUN source activate base
RUN conda install -y gpuci-tools; \
    conda env create --name cheminfo -f /tmp/env.yml; \
    sed -i 's/conda activate base/conda activate cheminfo/g' ~/.bashrc; \
    conda clean -afy && chmod -R ugo+w /opt/conda

# ADD source dest
# Create symlink for old scripts expecting `gdf` conda env
RUN ln -s /opt/conda/envs/cheminfo /opt/conda/envs/gdf

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
