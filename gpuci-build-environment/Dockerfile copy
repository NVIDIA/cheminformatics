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

ARG CC_VERSION=7
ARG PYTHON_VERSION=3.6
# Capture argument used for FROM
ARG CUDA_VER

# Update environment for gcc/g++ builds
ENV CC=/usr/bin/gcc
ENV CXX=/usr/bin/g++
ENV CUDAHOSTCXX=/usr/bin/g++
ENV CUDA_HOME=/usr/local/cuda
ENV LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/local/cuda/lib64:/usr/local/lib

# Enables "source activate conda"
SHELL ["/bin/bash", "-c"]

RUN apt update && \
    apt-get -y install  vim \
    # Install gcc version
    g++-${CC_VERSION} \
    gcc-${CC_VERSION} \
    cmake \
    make \
    # Install the packages needed to build with.
    # Install htslib dependencies
    wget \
    tabix \
    zlib1g-dev \
    libbz2-dev \
    liblzma-dev \
    libcurl4-gnutls-dev \
    # VariantWorks `cyvcf2` dependency
    libssl-dev \
    # samtools dependencies
    libncurses5-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

RUN update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-${CC_VERSION} 1000 --slave /usr/bin/g++ g++ /usr/bin/g++-${CC_VERSION}

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
RUN source activate base \
    && conda install -y gpuci-tools \
    && gpuci_conda_retry create --no-default-packages --override-channels -n parabricks \
      -c nvidia \
      -c conda-forge \
      -c defaults \
      -c gpuci \
      -c bioconda \
      autoconf \
      cudatoolkit=${CUDA_VER} \
      git \
      gpuci-tools \
      htslib \
      python=${PYTHON_VERSION} \
      rsync \
      "setuptools<50" \
    && sed -i 's/conda activate base/conda activate parabricks/g' ~/.bashrc ;

# Install samtools
RUN wget https://github.com/samtools/samtools/releases/download/1.12/samtools-1.12.tar.bz2 && \
    tar -xf samtools-1.12.tar.bz2  && \
    cd  samtools-1.12 && \
    ./configure && make && make install

# ADD source dest
# Create symlink for old scripts expecting `gdf` conda env
RUN ln -s /opt/conda/envs/parabricks /opt/conda/envs/gdf

# Clean up pkgs to reduce image size
RUN conda clean -afy && chmod -R ugo+w /opt/conda

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash" ]
