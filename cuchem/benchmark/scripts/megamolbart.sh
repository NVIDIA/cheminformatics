#!/usr/bin/env bash

SCRIPT_LOC=$(dirname "$0")

ID=100
ACTION="up"
GPU_ID="0"
MODEL_DIR="/models"
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
export RUN_ID="_${ID}"
export PLOTLY_PORT="5${ID}"

export SUBNET=192.${ID}.100.0/16
export IP_CUCHEM_UI=192.${ID}.100.1
export IP_MEGAMOLBART=192.${ID}.100.2

export CUCHEM_UI_START_CMD="python3 ./cuchem/cuchem/benchmark/megamolbart.py --config-dir /workspace/cuchem/benchmark/scripts"

export MEGAMOLBART_CMD="bash -c 'CUDA_VISIBLE_DEVICES=${GPU_ID} python3 launch.py'"

export CUCHEM_PATH=/workspace
export MEGAMOLBART_PATH=/workspace/megamolbart
export WORKSPACE_DIR="$(pwd)"

docker-compose \
  --env-file ${SCRIPT_LOC}/../../../.env  \
  -f ${SCRIPT_LOC}/../../../setup/docker_compose.yml \
  --project-directory ${SCRIPT_LOC}/../../../ \
  --project-name "megamolbart${RUN_ID}" \
  ${ACTION}
