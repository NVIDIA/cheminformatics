#!/usr/bin/env bash
#
# Copyright (c) 2020-2021, NVIDIA CORPORATION.
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


function version {
    echo "$@" | awk -F. '{ printf("%03d%03d%03d\n", $1,$2,$3); }';
}

function config() {
    local write_env=$1

    CUCHEM_CONT=${CUCHEM_CONT:=nvcr.io/nvidia/clara/cheminformatics_demo:0.1.2}
    MEGAMOLBART_CONT=${MEGAMOLBART_CONT:=nvcr.io/nvidia/clara/megamolbart:0.1.2}
    CDDD_CONT=${CDDD_CONT:=claraparabricks/cddd:1.0.0}
    MEGAMOLBART_MODEL=${MEGAMOLBART_MODEL:=nvidia/clara/megamolbart:0.1}
    PROJECT_PATH=${PROJECT_PATH:=$(pwd)}
    CONTENT_PATH=${CONTENT_PATH:=$(pwd)}
    DATA_MOUNT_PATH=${DATA_MOUNT_PATH:=/data}
    PLOTLY_PORT=${PLOTLY_PORT:-5000}
    DASK_PORT=${DASK_PORT:-9001}
    SUBNET=${SUBNET:=192.168.100.0/16}
    IP_CUCHEM_UI=${IP_CUCHEM_UI:=192.168.100.1}
    IP_MEGAMOLBART=${IP_MEGAMOLBART:=192.168.100.2}

    if [ $write_env -eq 1 ]; then
        echo CUCHEM_CONT=${CUCHEM_CONT} > $LOCAL_ENV
        echo MEGAMOLBART_CONT=${MEGAMOLBART_CONT} >> $LOCAL_ENV
        echo CDDD_CONT=${CDDD_CONT} >> $LOCAL_ENV
        echo MEGAMOLBART_MODEL=${MEGAMOLBART_MODEL} >> $LOCAL_ENV
        echo PROJECT_PATH=${PROJECT_PATH} >> $LOCAL_ENV
        echo CONTENT_PATH=${CONTENT_PATH} >> $LOCAL_ENV
        echo PLOTLY_PORT=${PLOTLY_PORT} >> $LOCAL_ENV
        echo DASK_PORT=${DASK_PORT} >> $LOCAL_ENV
        echo SUBNET=${SUBNET} >> $LOCAL_ENV
        echo IP_CUCHEM_UI=${IP_CUCHEM_UI} >> $LOCAL_ENV
        echo IP_MEGAMOLBART=${IP_MEGAMOLBART} >> $LOCAL_ENV
        echo REGISTRY=nvcr.io >> $LOCAL_ENV
        echo REGISTRY_USER="'\$oauthtoken'" >> $LOCAL_ENV
        echo REGISTRY_ACCESS_TOKEN= >> $LOCAL_ENV
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


validate_docker() {
    # Compare Docker version to find Nvidia Container Toolkit support.
    # Please refer https://github.com/NVIDIA/nvidia-docker
    PARAM_RUNTIME="--runtime=nvidia"
    DOCKER_VERSION_WITH_GPU_SUPPORT="19.03.0"
    if [ -x "$(command -v docker)" ]; then
        DOCKER_VERSION=$(docker version | grep -i version | head -1 | awk '{print $2'})
        if [ "$(version "$DOCKER_VERSION_WITH_GPU_SUPPORT")" -gt "$(version "$DOCKER_VERSION")" ]; then
            PARAM_RUNTIME="--gpus all"
        fi
    else
        if [[ ! -d "/opt/nvidia/cheminfomatics" ]]; then
            echo -e "${RED}${BOLD}Please install docker. https://docs.docker.com/engine/install/${RESET}."
            exit 1
        fi
    fi

    if [[ ! -d "/opt/nvidia/cheminfomatics" ]]; then
        DOCKER_COMPOSE_SUPPORTED="1.29.1"
        if [ -x "$(command -v docker-compose)" ]; then
            DOCKER_COMPOSE_VERSION=$(docker-compose version --short)
            if [ "$(version "$DOCKER_COMPOSE_SUPPORTED")" -gt "$(version "$DOCKER_COMPOSE_VERSION")" ]; then
                echo "${RED}${BOLD}Please upgrade docker-compose to ${DOCKER_COMPOSE_SUPPORTED} from https://docs.docker.com/compose/install/.${RESET}"
                exit 1
            fi
        else
            echo -e "${RED}${BOLD}Please install docker-compose. Version ${DOCKER_COMPOSE_SUPPORTED} or better. https://docs.docker.com/compose/install/${RESET}"
            exit 1
        fi
    fi
}

if [[ -d "/opt/nvidia/cheminfomatics" ]]; then
    DATA_PATH="/data"
    MODEL_PATH="/models"
else
    DATA_PATH="${CONTENT_PATH}/data"
    MODEL_PATH="${CONTENT_PATH}/models"
fi

DOCKER_CMD="docker run \
    --network host \
    --gpus all \
    ${PARAM_RUNTIME} \
    -p ${JUPYTER_PORT}:8888 \
    -p ${DASK_PORT}:${DASK_PORT} \
    -p ${PLOTLY_PORT}:5000 \
    -v ${PROJECT_PATH}:/workspace \
    -v ${DATA_PATH}:${DATA_MOUNT_PATH} \
    -v ${MODEL_PATH}:/models \
    --shm-size=1g \
    --ulimit memlock=-1 \
    --ulimit stack=67108864 \
    -e HOME=/workspace \
    -e TF_CPP_MIN_LOG_LEVEL=3"


dbSetup() {
    local DATA_DIR=$1

    if [[ ! -e "${DATA_DIR}/db/chembl_27.db" ]]; then
        echo -e "${YELLOW}Downloading chembl db to ${DATA_DIR}...${RESET}"
        mkdir -p ${DATA_DIR}/db
        if [[ ! -e "${DATA_DIR}/chembl_27_sqlite.tar.gz" ]]; then
            wget -q --show-progress \
                -O ${DATA_DIR}/chembl_27_sqlite.tar.gz \
                https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_27/chembl_27_sqlite.tar.gz
            return_code=$?
            if [[ $return_code -ne 0 ]]; then
                echo -e "${RED}${BOLD}ChEMBL DB download failed. Please check network settings, disk space(25GB) and wget version>=(1.20.3).${RESET}"
                rm -rf ${DATA_DIR}/chembl_27_sqlite.tar.gz
                exit $return_code
            fi
        fi

        wget -q --show-progress \
            -O ${DATA_DIR}/checksums.txt \
            https://ftp.ebi.ac.uk/pub/databases/chembl/ChEMBLdb/releases/chembl_27/checksums.txt
        echo "Unzipping chembl db to ${DATA_DIR}..."

        CURR_DIR=$PWD;
        if cd ${DATA_DIR}; sha256sum --check --ignore-missing --status ${DATA_DIR}/checksums.txt;
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
        cd ${CURR_DIR}
    else
        echo "${YELLOW}ChEMBL DB found at ${DATA_DIR}/db/chembl_27.db.${RESET}"
    fi
}


download_model() {
    set -e
    mkdir -p ${MODEL_PATH}
    local MEGAMOLBART_MODEL_VERSION=$(echo ${MEGAMOLBART_MODEL} | cut -d ":" -f2)
    echo -e "${YELLOW}Downloading model ${MEGAMOLBART_MODEL} to ${MODEL_PATH}...${RESET}"
    if [ -n "${ALT_MEGAMOLBART_MODEL}" ]; then
        # This is an alternate path for developers to download from an
        # alternate/pre-release location. Please add 'ALT_MEGAMOLBART_MODEL'
        # to .env with the alternate path. ALT_MEGAMOLBART_MODEL can only be
        # an NGC model and will require NGC installed and configured.
        local MEGAMOLBART_MODEL_VERSION=$(echo ${ALT_MEGAMOLBART_MODEL} | cut -d ":" -f2)

        if [[ ! -e "${MODEL_PATH}/megamolbart_v${MEGAMOLBART_MODEL_VERSION}" ]]; then
            local DOWNLOAD_URL=${MEGAMOLBART_MODEL_URL}
            mkdir -p ${MODEL_PATH}
            ngc registry model download-version \
                --dest ${MODEL_PATH} \
                "${ALT_MEGAMOLBART_MODEL}"
        fi
    elif [[ ! -e "${MODEL_PATH}/megamolbart_v${MEGAMOLBART_MODEL_VERSION}" ]]; then
        local DOWNLOAD_URL="https://api.ngc.nvidia.com/v2/models/nvidia/clara/megamolbart/versions/${MEGAMOLBART_MODEL_VERSION}/zip"

        wget -q --show-progress \
            --content-disposition ${DOWNLOAD_URL} \
            -O ${MODEL_PATH}/megamolbart_${MEGAMOLBART_MODEL_VERSION}.zip
        mkdir ${MODEL_PATH}/megamolbart_v${MEGAMOLBART_MODEL_VERSION}
        unzip -q ${MODEL_PATH}/megamolbart_${MEGAMOLBART_MODEL_VERSION}.zip \
            -d ${MODEL_PATH}/megamolbart_v${MEGAMOLBART_MODEL_VERSION}
    fi

    ln -nsf ${MODEL_PATH}/megamolbart_v${MEGAMOLBART_MODEL_VERSION} ${MODEL_PATH}/megamolbart
    set +e
}
