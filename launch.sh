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
BLACK=`tput setaf 0`
RED=`tput setaf 1`
GREEN=`tput setaf 2`
YELLOW=`tput setaf 3`
BLUE=`tput setaf 4`
MAGENTA=`tput setaf 5`
CYAN=`tput setaf 6`
WHITE=`tput setaf 7`

BOLD=`tput bold`
RESET=`tput sgr0`

LOCAL_ENV=.env

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
    ./launch config
    ./launch build
    ./launch start
    navigate browser to http://localhost:5000
For more detailed info on getting started, see README.md


More Information
----------------------------------------

Note: This script looks for a file called $LOCAL_ENV in the
current directory. This file should define the following environment
variables:
    CONT_NAME
        container image, prepended with registry. e.g.,
        benchmark:latest
    PROJECT_PATH
        path to source code. e.g.,
        /home/user/projects/cheminformatics
    MODEL_PATH
        path to location of all models. e.g.,
        /home/user/projects/cheminformatics/models
    DATA_PATH
        path to data directory. e.g.,
        /scratch/data/cheminformatics/data

EOF
}


function config() {
    local write_env=$1

    IMG_NAME=${IMG_NAME:=nvcr.io/nvidia/clara/benchmark:0.1.2}
    PROJECT_PATH=${PROJECT_PATH:=$(pwd)}
    DATA_PATH=${DATA_PATH:=$(pwd)/data}
    MODEL_PATH=${MODEL_PATH:=$(pwd)/models}
    JUPYTER_PORT=${JUPYTER_PORT:=8888}

    REPO_BRANCH=${REPO_BRANCH:=master}

    REGISTRY=${REGISTRY:=nvcr.io}
    REGISTRY_USER=${REGISTRY_USER:='$oauthtoken'}
    REGISTRY_ACCESS_TOKEN=${REGISTRY_ACCESS_TOKEN:=NotSet}


    if [ $write_env -eq 1 ]; then
        echo IMG_NAME=${IMG_NAME} > $LOCAL_ENV
        echo PROJECT_PATH=${PROJECT_PATH} >> $LOCAL_ENV
        echo MODEL_PATH=${MODEL_PATH} >> $LOCAL_ENV
        echo DATA_PATH=${DATA_PATH} >> $LOCAL_ENV
        echo JUPYTER_PORT=${JUPYTER_PORT} >> $LOCAL_ENV

        echo REPO_BRANCH=${REPO_BRANCH} >> $LOCAL_ENV

        echo REGISTRY=${REGISTRY} >> $LOCAL_ENV
        echo REGISTRY_USER=${REGISTRY_USER} >> $LOCAL_ENV
        echo REGISTRY_ACCESS_TOKEN=${REGISTRY_ACCESS_TOKEN} >> $LOCAL_ENV

    fi
}

if [ -e ./${LOCAL_ENV} ]
then
    echo -e "sourcing environment from ./${LOCAL_ENV}"
    . ./${LOCAL_ENV}
    config 0
else
    echo -e "${YELLOW}Writing deafults to ${LOCAL_ENV}${RESET}"
    config 1
fi

CONT_NAME='chem_benchmark'
WORKSPACE_DIR='/workspace'
DOCKER_CMD="docker run \
    --network host \
    --gpus all \
    -p ${JUPYTER_PORT}:8888 \
    -v ${DATA_PATH}:/data \
    -v ${MODEL_PATH}:/models \
    -e HOME=/workspace"


build() {
    local IMG_OPTION=$1
    set -e
    IFS=':' read -ra IMG_NAME_BASENAME <<< ${IMG_NAME}
    echo "Building ${IMG_NAME_BASENAME}..."
    docker build --network host \
        --build-arg REPO_BRANCH=${REPO_BRANCH} \
        -t ${IMG_NAME_BASENAME}:latest \
        -t ${IMG_NAME} \
        -f Dockerfile .

    set +e
}


push() {
    local container_name=($(echo ${IMG_NAME} | tr ":" "\n"))

    docker login ${REGISTRY} -u ${REGISTRY_USER} -p ${REGISTRY_ACCESS_TOKEN}
    docker push ${container_name[0]}:latest
    docker tag ${container_name[0]}:latest ${container_name[0]}:${container_name[1]}
    docker push ${container_name[0]}:${container_name[1]}
    exit
}


dev() {
    local DEV_IMG=${IMG_NAME}
    local CMD="bash"

    DOCKER_CMD="${DOCKER_CMD} --name ${CONT_NAME}"
    DOCKER_CMD="${DOCKER_CMD} -w /workspace/"
    DOCKER_CMD="${DOCKER_CMD} -v ${DATA_PATH}/logs/:/logs"
    DOCKER_CMD="${DOCKER_CMD} -v ${PROJECT_PATH}:${WORKSPACE_DIR}/cheminformatics"
    PYTHONPATH="${PYTHONPATH}:${WORKSPACE_DIR}/cheminformatics/benchmark"

    if [ ! -z "${NEMO_SOURCE_PATH}" ];
    then
        DOCKER_CMD="${DOCKER_CMD} -v ${NEMO_SOURCE_PATH}:${WORKSPACE_DIR}/nemo"
        PYTHONPATH="${PYTHONPATH}:${WORKSPACE_DIR}/nemo"
    fi

    if [ ! -z "${NEMO_CHEM_SOURCE_PATH}" ];
    then
        DOCKER_CMD="${DOCKER_CMD} -v ${NEMO_CHEM_SOURCE_PATH}:${WORKSPACE_DIR}/nemo_chem"
        PYTHONPATH="${PYTHONPATH}:${WORKSPACE_DIR}/nemo_chem"
    fi
    DOCKER_CMD="${DOCKER_CMD} -e PYTHONPATH=${PYTHONPATH}"

    while [[ $# -gt 0 ]]; do
        case $1 in
            -a|--additional-args)
                DOCKER_CMD="${DOCKER_CMD} $2"
                shift
                shift
                ;;
            -i|--image)
                DEV_IMG="$2"
                shift
                shift
                ;;
            -d)
                DOCKER_CMD="${DOCKER_CMD} -d"
                shift
                ;;
            -c|--cmd)
                CMD=$2
                shift
                shift
                ;;
            *)
                echo "Unknown option $1"
                exit 1
                ;;
        esac
    done

    ${DOCKER_CMD} -it --rm ${DEV_IMG} ${CMD}
}


attach() {
    DOCKER_CMD="docker exec"
    CONTAINER_ID=$(docker ps | grep ${CONT_NAME} | cut -d' ' -f1)
    ${DOCKER_CMD} -it ${CONTAINER_ID} /bin/bash
    exit
}


case $1 in
    build)
        "$@"
        ;;
    config)
        config 1
        ;;
    push)
        ;&
    attach)
        $@
        ;;
    dev)
        "$@"
        ;;
    *)
        usage
        ;;
esac
