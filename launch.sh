#!/bin/bash
#
# Copyright (c) 2020, NVIDIA CORPORATION.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

###############################################################################
#
# This is my $LOCAL_ENV file
#
LOCAL_ENV=.cheminf_local_environment
#
###############################################################################

usage() {
    cat <<EOF

USAGE: launch.sh

launch utility script
----------------------------------------

launch.sh [command]

    valid commands:
        start
        stop
        build


Getting Started tl;dr
----------------------------------------

    ./launch build
    ./launch dash
    navigate browser to http://localhost:5000
For more detailed info on getting started, see README.md


More Information
----------------------------------------

Note: This script looks for a file called $LOCAL_ENV in the
current directory. This file should define the following environment
variables:
    CUCHEM_CONT
        container image, prepended with registry. e.g.,
        cheminformatics_demo:latest
    MEGAMOLBART_TRAINING_CONT
        container image for MegaMolBART training, prepended with registry. e.g.,
        Note that this is a separate (precursor) container from any service associated containers
    MEGAMOLBART_SERVICE_CONT
        container image for MegaMolBART service, prepended with registry.
    PROJECT_PATH
        path to repository. e.g.,
        /home/user/projects/cheminformatics
    DATA_PATH
        path to data directory. e.g.,
        /scratch/data/cheminformatics
    REGISTRY_ACCESS_TOKEN
        container registry access token. e.g.,
        Ckj53jGK...
    REGISTRY_USER
        container registry username. e.g.,
        astern
    REGISTRY
        container registry URL. e.g.,
        server.com/registry:5005

EOF
    exit
}


###############################################################################
#
# if $LOCAL_ENV file exists, source it to specify my environment
#
###############################################################################

if [ -e ./$LOCAL_ENV ]
then
    echo sourcing environment from ./$LOCAL_ENV
    . ./$LOCAL_ENV
    write_env=0
else
    echo $LOCAL_ENV does not exist. Writing deafults to $LOCAL_ENV
    write_env=1
fi

###############################################################################
#
# alternatively, override variable here.  These should be all that are needed.
#
###############################################################################

CUCHEM_CONT=${CUCHEM_CONT:=nvcr.io/nvidia/clara/cheminformatics_demo:0.2}
MEGAMOLBART_TRAINING_CONT=${MEGAMOLBART_TRAINING_CONT:=nvcr.io/nvidian/clara-lifesciences/megamolbart_training:latest}
MEGAMOLBART_SERVICE_CONT=${MEGAMOLBART_SERVICE_CONT:=nvcr.io/nvidian/clara-lifesciences/megamolbart:latest}
PROJECT_PATH=${PROJECT_PATH:=$(pwd)}
DATA_PATH=${DATA_PATH:=/tmp}
DATA_MOUNT_PATH=${DATA_MOUNT_PATH:=/data}
JUPYTER_PORT=${JUPYTER_PORT:-9000}
PLOTLY_PORT=${PLOTLY_PORT:-5000}
DASK_PORT=${DASK_PORT:-9001}
GITHUB_ACCESS_TOKEN=${GITHUB_ACCESS_TOKEN:=""}

DEV_PYTHONPATH="/workspace/cuchem:/workspace/common:/workspace/common/generated/"
###############################################################################
#
# If $LOCAL_ENV was not found, write out a template for user to edit
#
###############################################################################

if [ $write_env -eq 1 ]; then
    echo CUCHEM_CONT=${CUCHEM_CONT} >> $LOCAL_ENV
    echo MEGAMOLBART_CONT=${MEGAMOLBART_CONT} >> $LOCAL_ENV
    echo PROJECT_PATH=${PROJECT_PATH} >> $LOCAL_ENV
    echo DATA_PATH=${DATA_PATH} >> $LOCAL_ENV
    echo DATA_MOUNT_PATH=${DATA_MOUNT_PATH} >> $LOCAL_ENV
    echo JUPYTER_PORT=${JUPYTER_PORT} >> $LOCAL_ENV
    echo PLOTLY_PORT=${PLOTLY_PORT} >> $LOCAL_ENV
    echo DASK_PORT=${DASK_PORT} >> $LOCAL_ENV
fi

###############################################################################
#
#          shouldn't need to make changes beyond this point
#
###############################################################################
# Compare Docker version to find Nvidia Container Toolkit support.
# Please refer https://github.com/NVIDIA/nvidia-docker
DOCKER_VERSION_WITH_GPU_SUPPORT="19.03.0"
if [ -x "$(command -v docker)" ]; then
    DOCKER_VERSION=$(docker version | grep -i version | head -1 | awk '{print $2'})
fi

if [ -e /workspace/cuchem/startdash.py ]; then
    # When inside container in dev mode
    CUCHEM_LOC="/workspace/cuchem/"
elif [ -e /opt/nvidia/cheminfomatics/cuchem/startdash.py ]; then
    # When inside container in prod mode
    CUCHEM_LOC="/opt/nvidia/cheminfomatics/cuchem/"
else
    # On baremetal
    CUCHEM_LOC="./"
fi

PARAM_RUNTIME="--runtime=nvidia"
if [ "$DOCKER_VERSION_WITH_GPU_SUPPORT" == "$(echo -e "$DOCKER_VERSION\n$DOCKER_VERSION_WITH_GPU_SUPPORT" | sort -V | head -1)" ];
then
    PARAM_RUNTIME="--gpus all"
fi

DOCKER_CMD="docker run \
    --rm \
    --network host \
    ${PARAM_RUNTIME} \
    -p ${JUPYTER_PORT}:8888 \
    -p ${DASK_PORT}:${DASK_PORT} \
    -p ${PLOTLY_PORT}:5000 \
    -v ${PROJECT_PATH}:/workspace \
    -v ${DATA_PATH}:${DATA_MOUNT_PATH} \
    -u $(id -u ${USER}):$(id -g ${USER}) \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e HOME=/workspace \
    -e TF_CPP_MIN_LOG_LEVEL=3 \
    -w /workspace"


build() {
    set -e
    DATE=$(date +%y%m%d)

    IFS=':' read -ra CUCHEM_CONT_BASENAME <<< ${CUCHEM_CONT}
    echo "Building ${CUCHEM_CONT_BASENAME}..."
    docker build --network host \
        -t ${CUCHEM_CONT_BASENAME}:latest \
        -t ${CUCHEM_CONT_BASENAME}:${DATE} \
        -f Dockerfile.cuchem .

    IFS=':' read -ra MEGAMOLBART_CONT_BASENAME <<< ${MEGAMOLBART_SERVICE_CONT}
    echo "Building ${MEGAMOLBART_CONT_BASENAME}..."
    docker build --no-cache --network host \
        -t ${MEGAMOLBART_CONT_BASENAME}:latest \
        -t ${MEGAMOLBART_CONT_BASENAME}:${DATE} \
        --build-arg SOURCE_CONTAINER=${MEGAMOLBART_TRAINING_CONT} \
        -f Dockerfile.megamolbart \
        .

    set +e
    exit
}


push() {
    docker login ${REGISTRY} -u ${REGISTRY_USER} -p ${REGISTRY_ACCESS_TOKEN}
    docker push ${CUCHEM_CONT}:latest
    docker push ${CUCHEM_CONT}:${DATE}
    docker push ${MEGAMOLBART_SERVICE_CONT}:latest
    docker push ${MEGAMOLBART_SERVICE_CONT}:${DATE}
    exit
}


pull() {
    docker login ${REGISTRY} -u ${REGISTRY_USER} -p ${REGISTRY_ACCESS_TOKEN}
    docker pull ${CUCHEM_CONT}
    docker pull ${MEGAMOLBART_SERVICE_CONT}
    exit
}


dev() {
    set -x
    local CONTAINER_OPTION=$1
    local CONT=${CUCHEM_CONT}

    if [[ ${CONTAINER_OPTION} -eq 2 ]]; then
        DOCKER_CMD="${DOCKER_CMD} -v ${PROJECT_PATH}/megamolbart/checkpoints/megatron:/models/megamolbart/checkpoints"
        CONT=${MEGAMOLBART_SERVICE_CONT}
    else
        DOCKER_CMD="${DOCKER_CMD} -e PYTHONPATH=${DEV_PYTHONPATH}"
    fi

    ${DOCKER_CMD} -it ${CONT} bash

    exit
}


dbSetup() {
    local DATA_DIR=$1

    if [[ ! -e "${DATA_DIR}/db/chembl_27.db" ]]; then
        echo "Downloading chembl db to ${DATA_DIR}..."
        mkdir -p ${DATA_DIR}/db
        if [[ ! -e "${DATA_DIR}/chembl_27_sqlite.tar.gz" ]]; then
            wget -q --show-progress \
                -O ${DATA_DIR}/chembl_27_sqlite.tar.gz \
                ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_27/chembl_27_sqlite.tar.gz
            return_code=$?
            if [[ $return_code -ne 0 ]]; then
                echo 'ChEMBL database download failed. Please check network settings.'
                rm -rf ${DATA_DIR}/chembl_27_sqlite.tar.gz
                exit $return_code
            fi
        fi

        wget -q --show-progress \
            -O ${DATA_DIR}/checksums.txt \
            ftp://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_27/checksums.txt
        echo "Unzipping chembl db to ${DATA_DIR}..."
        if cd ${DATA_DIR}; sha256sum --check --ignore-missing --status ${DATA_DIR}/checksums.txt
        then
            tar -C ${DATA_DIR}/db \
                --strip-components=2 \
                -xf ${DATA_DIR}/chembl_27_sqlite.tar.gz chembl_27/chembl_27_sqlite/chembl_27.db
            return_code=$?
            if [[ $return_code -ne 0 ]]; then
                echo 'ChEMBL database extraction faile. Please cleanup ${DATA_DIR} directory and retry.'
                rm -rf ${DATA_DIR}/chembl_27_sqlite.tar.gz
                exit $return_code
            fi
        else
            echo "Please clean ${DATA_DIR} directory and retry."
            exit 1
        fi
    fi
}


start() {
    if [[ -d "/opt/nvidia/cheminfomatics" ]]; then
        # Executed within container or a managed env.
        dbSetup "${DATA_MOUNT_PATH}"
        cd ${CUCHEM_LOC}; python3 ${CUCHEM_LOC}/startdash.py analyze $@
    else
        # run a container and start dash inside container.
        echo "${CUCHEM_CONT} ${MEGAMOLBART_CONT}"
        export ADDITIONAL_PARAM="$@"
        docker-compose --env-file .cheminf_local_environment  \
            -f setup/docker_compose.yml \
            --project-directory . \
            up
    fi
    exit
}


stop() {
    docker-compose --env-file .cheminf_local_environment  \
        -f setup/docker_compose.yml \
        --project-directory . \
        down
}


cache() {
    if [[ -d "/opt/nvidia/cheminfomatics" ]]; then
        set -x
        # Executed within container or a managed env.
        dbSetup "${DATA_MOUNT_PATH}"
        cd ${CUCHEM_LOC}; python3 startdash.py cache $@
    else
        dbSetup "${DATA_PATH}"
        # run a container and start dash inside container.
        ${DOCKER_CMD} -it ${CUCHEM_CONT} ./launch.sh cache $@
    fi
    exit
}


test() {
    dbSetup "${DATA_PATH}"
    # run a container and start dash inside container.
    ${DOCKER_CMD} -w /workspace/cuchem \
        -e PYTHONPATH="${DEV_PYTHONPATH}" \
        ${CUCHEM_CONT}  \
        pytest tests
    exit
}


jupyter() {
    ${DOCKER_CMD} -it ${CUCHEM_CONT} jupyter-lab --no-browser \
        --port=8888 \
        --ip=0.0.0.0 \
        --notebook-dir=/workspace \
        --NotebookApp.password=\"\" \
        --NotebookApp.token=\"\" \
        --NotebookApp.password_required=False
    exit
}


case $1 in
    build)
        ;&
    push)
        ;&
    pull)
        ;&
    dev)
        $@
        ;;
    test)
        ;&
    start)
        $@
        ;;
    stop)
        ;&
    cache)
        $@
        ;;
    jupyter)
        $1
        ;;
    *)
        usage
        ;;
esac
