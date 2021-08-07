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

source setup/env.sh
MEGAMOLBART_TRAINING_CONT=${MEGAMOLBART_TRAINING_CONT:=nvcr.io/nvidian/clara-lifesciences/megamolbart_training:latest}
DEV_PYTHONPATH="/workspace/cuchem:/workspace/common:/workspace/common/generated/"

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

build() {
    set -e
    DATE=$(date +%y%m%d)

    IFS=':' read -ra CUCHEM_CONT_BASENAME <<< ${CUCHEM_CONT}
    echo "Building ${CUCHEM_CONT_BASENAME}..."
    docker build --network host \
        -t ${CUCHEM_CONT_BASENAME}:latest \
        -t ${CUCHEM_CONT_BASENAME}:${DATE} \
        -f Dockerfile.cuchem .

    IFS=':' read -ra MEGAMOLBART_CONT_BASENAME <<< ${MEGAMOLBART_CONT}
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
    local VERSION=$1
    set -x
    IFS=':' read -ra CUCHEM_CONT_BASENAME <<< ${CUCHEM_CONT}
    IFS=':' read -ra MEGAMOLBART_BASENAME <<< ${MEGAMOLBART_CONT}

    docker login ${REGISTRY} -u ${REGISTRY_USER} -p ${REGISTRY_ACCESS_TOKEN}

    docker push ${CUCHEM_CONT_BASENAME}:latest
    docker tag ${CUCHEM_CONT_BASENAME}:latest ${CUCHEM_CONT_BASENAME}:${VERSION}
    docker push ${CUCHEM_CONT_BASENAME}:${VERSION}

    docker push ${MEGAMOLBART_BASENAME}:latest
    docker tag ${MEGAMOLBART_BASENAME}:latest ${MEGAMOLBART_BASENAME}:${VERSION}
    docker push ${MEGAMOLBART_BASENAME}:${VERSION}
    exit
}


pull() {
    docker login ${REGISTRY} -u ${REGISTRY_USER} -p ${REGISTRY_ACCESS_TOKEN}
    docker pull ${CUCHEM_CONT}
    docker pull ${MEGAMOLBART_CONT}
    exit
}


dev() {
    local CONTAINER_OPTION=$1
    local CONT=${CUCHEM_CONT}

    if [[ ${CONTAINER_OPTION} -eq 2 ]]; then
        DOCKER_CMD="${DOCKER_CMD} -v ${CONTENT_PATH}/models/megamolbart_v0.1/:/models/megamolbart/"
        DOCKER_CMD="${DOCKER_CMD} -w /workspace/megamolbart/"
        DOCKER_CMD="${DOCKER_CMD} -e PYTHONPATH=${DEV_PYTHONPATH}:/workspace/megamolbart:/opt/MolBART/megatron_molbart/"
        CONT=${MEGAMOLBART_CONT}
    else
        DOCKER_CMD="${DOCKER_CMD} --privileged"
        DOCKER_CMD="${DOCKER_CMD} -v ${PROJECT_PATH}/chemportal/config:/etc/nvidia/cuChem/"
        DOCKER_CMD="${DOCKER_CMD} -v /var/run/docker.sock:/var/run/docker.sock"
        DOCKER_CMD="${DOCKER_CMD} -e PYTHONPATH=${DEV_PYTHONPATH}:"
        DOCKER_CMD="${DOCKER_CMD} -w /workspace/chemportal/"
    fi

    ${DOCKER_CMD} -it ${CONT} bash

    exit
}


start() {
    if [[ -d "/opt/nvidia/cheminfomatics" ]]; then
        # Executed within container or a managed env.
        if [[ -d "/workspace/common/generated" ]]; then
            PYTHONPATH="/workspace/cuchem:/workspace/common:/workspace/common/generated/"
        fi
        dbSetup "${DATA_MOUNT_PATH}"
        cd ${CUCHEM_LOC}; python3 ${CUCHEM_LOC}/startdash.py analyze $@
    else
        # run a container and start dash inside container.
        download_model
        dbSetup "${DATA_PATH}"

        export CUCHEM_UI_START_CMD="./launch.sh start $@"
        export MEGAMOLBART_CMD="python3 launch.py"
        export CUCHEM_PATH=/workspace
        export MEGAMOLBART_PATH=/workspace/megamolbart
        docker-compose --env-file .env  \
                -f setup/docker_compose.yml \
                --project-directory . \
                up
    fi
    exit
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
        $@
        ;;
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