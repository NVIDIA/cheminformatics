#!/usr/bin/env bash
set -e

SCRIPT_LOC=$(dirname "$0")

ID=100
ACTION="up"
GPU_ID="0"
MODEL_DIR="/models"
CONFIG_DIR="/workspace/benchmark/scripts"
SIZE=''
NUM_LAYERS=4
HIDDEN_SIZE=256
NUM_ATTENTION_HEADS=8

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
  --id)
    ID=$2
    shift
    shift
    ;;
  --gpu)
    GPU_ID=$2
    shift
    shift
    ;;
  --stop)
    ACTION=stop
    shift
    shift
    ;;
  --ckp)
    MODEL_DIR=$2
    shift
    shift
    ;;
  --config-dir)
    CONFIG_DIR=$2
    shift
    shift
    ;;
  --size)
    SIZE=$2
    shift
    shift
    ;;
  *)
    shift
    ;;
  esac
done
source ${SCRIPT_LOC}/.env
export RUN_ID="_${ID}"
export PLOTLY_PORT="5${ID}"

export SUBNET=192.${ID}.100.0/16
export IP_CUCHEM_UI=192.${ID}.100.1
export IP_MEGAMOLBART=192.${ID}.100.2

export CUCHEM_UI_START_CMD="python3 -m cuchembm --config-dir ${CONFIG_DIR}"

export MEGAMOLBART_CMD="bash -c 'CUDA_VISIBLE_DEVICES=${GPU_ID} python3 -m megamolbart'"

CHEMINFO_DIR='/workspace'
export PYTHONPATH_CUCHEM="${CHEMINFO_DIR}/benchmark:${CHEMINFO_DIR}/cuchem:${CHEMINFO_DIR}/common:/${CHEMINFO_DIR}/common/generated/"
export PYTHONPATH_MEGAMOLBART="${CHEMINFO_DIR}/common:/${CHEMINFO_DIR}/common/generated/"

export WORKING_DIR_CUCHEMUI=/workspace
export WORKING_DIR_MEGAMOLBART=/workspace/megamolbart
# export UID=$(id -u)
export GID=$(id -g)

docker-compose \
  -f ${SCRIPT_LOC}/setup/docker_compose.yml \
  --project-directory ${SCRIPT_LOC}/ \
  --project-name "megamolbart${RUN_ID}" \
  ${ACTION}
