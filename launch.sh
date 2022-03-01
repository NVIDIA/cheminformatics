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
    CUCHEM_CONT
        container image, prepended with registry. e.g.,
        cheminformatics_demo:latest
    MEGAMOLBART_CONT
        container image for MegaMolBART service, prepended with registry.
    MEGAMOLBART_MODEL
        MegaMolBART model and the version to use.
    CONTENT_PATH
        path to repository. e.g.,
        /home/user/projects/cheminformatics
    DATA_PATH
        path to data directory. e.g.,
        /scratch/data/cheminformatics

EOF
}

source setup/env.sh
CHEMINFO_DIR='/workspace'
if [ -e /workspace/cuchem/startdash.py ]; then
    # When inside container in dev/test mode
    CHEMINFO_DIR='/workspace'
elif [ -e /opt/nvidia/cheminfomatics/cuchem/startdash.py ]; then
    # When inside container in prod mode
    CHEMINFO_DIR="/opt/nvidia/cheminfomatics"
fi
PYTHONPATH_CUCHEM="${CHEMINFO_DIR}/cuchem:${CHEMINFO_DIR}/common:${CHEMINFO_DIR}/common/generated/"

build() {
    local IMG_OPTION=$1
    set -e
    DATE=$(date +%y%m%d)

    local built=false
    if [[ -z "${IMG_OPTION}" || "${IMG_OPTION}" == "1" ]]; then
        IFS=':' read -ra CUCHEM_CONT_BASENAME <<< ${CUCHEM_CONT}
        echo "Building ${CUCHEM_CONT_BASENAME}..."
        docker build --network host \
            -t ${CUCHEM_CONT_BASENAME}:latest \
            -t ${CUCHEM_CONT} \
            -f Dockerfile.cuchem .
        built=true
    fi

    if [[ -z "${IMG_OPTION}" || "${IMG_OPTION}" == "2" ]]; then
        IFS=':' read -ra MEGAMOLBART_CONT_BASENAME <<< ${MEGAMOLBART_CONT}
        echo "Building ${MEGAMOLBART_CONT_BASENAME}..."
        docker build --network host \
            --build-arg GITHUB_ACCESS_TOKEN=${GITHUB_ACCESS_TOKEN} \
            -t ${MEGAMOLBART_CONT_BASENAME}:latest \
            -t ${MEGAMOLBART_CONT} \
            -f Dockerfile.megamolbart .
        built=true
    fi

    if [[ -z "${IMG_OPTION}" || "${IMG_OPTION}" == "3" ]]; then
        IFS=':' read -ra CDDD_CONT_BASENAME <<< ${CDDD_CONT}
        echo "Building ${CDDD_CONT_BASENAME}..."
        docker build --network host \
            -t ${CDDD_CONT_BASENAME}:latest \
            -t ${CDDD_CONT} \
            -f Dockerfile.cddd .
        built=true
    fi

    if [[ ${built} == false ]]; then
        echo "${RED}${BOLD}Invalid build option: ${IMG_OPTION}${RESET}"
        exit 1
    fi

    set +e
}


check_docker_registry() {
    if [ -z ${REGISTRY_ACCESS_TOKEN} ]; then
        echo "${RED}Please ensure 'REGISTRY_ACCESS_TOKEN' in $LOCAL_ENV is correct and rerun this script. Please set NGC API key to REGISTRY_ACCESS_TOKEN.${RESET}"
        exit 1
    else
        echo "${YELLOW}Attempting docker login to ${REGISTRY}.${RESET}"
    fi

    docker login ${REGISTRY} -u ${REGISTRY_USER} -p ${REGISTRY_ACCESS_TOKEN}
    if [[ $? -ne 0 ]]; then
        echo "${RED}Docker login failed. Please setup ngc('ngc config set'). "
        echo "Please also check network settings and ensure 'REGISTRY_ACCESS_TOKEN' is $LOCAL_ENV is correct.${RESET}"
        exit 1
    fi
}


push_container() {
    local container_name=($(echo $1 | tr ":" "\n"))
    docker push ${container_name[0]}:latest
    docker tag ${container_name[0]}:latest ${container_name[0]}:${container_name[1]}
    docker push ${container_name[0]}:${container_name[1]}
}


push() {
    local PUSH_OPTION=$1
    local pushed=false

    if [[ -z "${PUSH_OPTION}" || "${PUSH_OPTION}" == "1" ]]; then
        docker login ${REGISTRY} -u ${REGISTRY_USER} -p ${REGISTRY_ACCESS_TOKEN}
        push_container ${CUCHEM_CONT}
        pushed=true
    fi

    if [[ -z "${PUSH_OPTION}" || "${PUSH_OPTION}" == "2" ]]; then
        docker login ${REGISTRY} -u ${REGISTRY_USER} -p ${REGISTRY_ACCESS_TOKEN}
        push_container ${MEGAMOLBART_CONT}
        pushed=true
    fi

    if [[ -z "${PUSH_OPTION}" || "${PUSH_OPTION}" == "3" ]]; then
        push_container ${CDDD_CONT}
        pushed=true
    fi

    if [[ ${pushed} == false ]]; then
        echo "${RED}${BOLD}Invalid push option: ${IMG_OPTION}${RESET}"
        exit 1
    fi

    exit
}


setup() {
    download_model
    dbSetup "${DATA_PATH}"
}

dev() {
    local CONTAINER_OPTION=$1
    local CONT=${CUCHEM_CONT}

    if [[ ${CONTAINER_OPTION} -eq 1 ]]; then
        DOCKER_CMD="${DOCKER_CMD} --privileged"
        DOCKER_CMD="${DOCKER_CMD} -v ${PROJECT_PATH}/chemportal/config:/etc/nvidia/cuChem/"
        DOCKER_CMD="${DOCKER_CMD} -v ${CONTENT_PATH}/logs/:/logs"
        DOCKER_CMD="${DOCKER_CMD} -v /var/run/docker.sock:/var/run/docker.sock"
        DOCKER_CMD="${DOCKER_CMD} -e PYTHONPATH=${PYTHONPATH_CUCHEM}:/workspace/benchmark"
        DOCKER_CMD="${DOCKER_CMD} -w /workspace/cuchem/"
    elif [[ ${CONTAINER_OPTION} -eq 2 ]]; then
        DOCKER_CMD="${DOCKER_CMD} -v ${CONTENT_PATH}/models/megamolbart_v0.1/:/models/megamolbart/"
        DOCKER_CMD="${DOCKER_CMD} -v ${CONTENT_PATH}/logs/:/logs"
        DOCKER_CMD="${DOCKER_CMD} -v /var/run/docker.sock:/var/run/docker.sock"
        DOCKER_CMD="${DOCKER_CMD} -w /workspace/megamolbart/"
        DOCKER_CMD="${DOCKER_CMD} -e PYTHONPATH=${PYTHONPATH_CUCHEM}:/workspace/megamolbart:/workspace/benchmark"
        CONT=${MEGAMOLBART_CONT}
    elif [[ ${CONTAINER_OPTION} -eq 3 ]]; then
        DOCKER_CMD="${DOCKER_CMD} -v ${CONTENT_PATH}/logs/:/logs"
        DOCKER_CMD="${DOCKER_CMD} -w /workspace/"
        DOCKER_CMD="${DOCKER_CMD} -e PYTHONPATH=${PYTHONPATH_CUCHEM}:/workspace/cddd"
        CONT=${CDDD_CONT}
    else
        echo "${RED}${BOLD}Invalid container option${RESET}"
        exit 1
    fi

    if [ ! -z "$2" ]; then
        DOCKER_CMD="${DOCKER_CMD} -d"
        CMD="$2"
    else
        DOCKER_CMD="${DOCKER_CMD} --rm"
        CMD='bash'
    fi
    set -x
    ${DOCKER_CMD} -it ${CONT} ${CMD}
}


start() {
    validate_docker

    if [[ -d "/opt/nvidia/cheminfomatics" ]]; then
        PYTHONPATH=${PYTHONPATH_CUCHEM}
        dbSetup "${DATA_MOUNT_PATH}"
        cd ${CHEMINFO_DIR}/cuchem/; python3 startdash.py analyze $@
    else
        # run a container and start dash inside container.
        setup
        set -x
        export CUCHEM_UI_START_CMD="./launch.sh start $@"
        export UID=$(id -u)
        export GID=$(id -g)

        # Working directory for the individual containers.
        echo "Starting containers ${MEGAMOLBART_CONT} and ${CUCHEM_CONT}..."
        export WORKING_DIR_CUCHEMUI=/workspace
        export WORKING_DIR_MEGAMOLBART=/workspace/megamolbart
        export WORKING_DIR_CDDD=/workspace
        export PYTHONPATH_MEGAMOLBART="${CHEMINFO_DIR}/common:/${CHEMINFO_DIR}/common/generated/"
        export NGINX_CONFIG=${PROJECT_PATH}/setup/config/nginx.conf

        docker-compose --env-file .env  \
                -f setup/docker_compose.yml \
                --project-directory . \
                up
    fi
}


stop() {
    docker-compose --env-file .env  \
        -f setup/docker_compose.yml \
        --project-directory . \
        down
}


cache() {
    if [[ -d "/opt/nvidia/cheminfomatics" ]]; then
        set -x
        # Executed within container or a managed env.
        dbSetup "${DATA_MOUNT_PATH}"a601b2a2a627
        cd ${CHEMINFO_DIR}/cuchem/; python3 startdash.py cache $@
    else
        dbSetup "${DATA_PATH}"
        # run a container and start dash inside container.
        ${DOCKER_CMD} -it ${CUCHEM_CONT} ./launch.sh cache $@
    fi
}


test() {
    dbSetup "${DATA_PATH}"
    # run a container and start dash inside container.
    if [[ -d "/opt/nvidia/cheminfomatics" ]]; then
        pytest tests
    else
        ${DOCKER_CMD} -w /workspace/cuchem \
            -e PYTHONPATH="${PYTHONPATH_CUCHEM}" \
            ${CUCHEM_CONT}  \
            pytest tests
    fi
}


jupyter() {
    ${DOCKER_CMD} -it ${CUCHEM_CONT} jupyter-lab --no-browser \
        --port=8888 \
        --ip=0.0.0.0 \
        --allow-root \
        --notebook-dir=/workspace \
        --NotebookApp.password='' \
        --NotebookApp.token='' \
        --NotebookApp.password_required=False
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
    pull)
        ;&
    setup)
        ;&
    dev)
        "$@"
        ;;
    test)
        ;&
    start)
        "$@"
        ;;
    stop)
        ;&
    cache)
        "$@"
        ;;
    jupyter)
        $1
        ;;
    *)
        usage
        ;;
esac
