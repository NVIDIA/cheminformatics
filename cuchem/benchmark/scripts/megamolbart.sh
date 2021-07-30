#!/usr/bin/env bash

ID=100
ACTION="up"
GPU_ID="0"
CHECKPOINT_DIR="/models/megamolbart/checkpoints"

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
    CHECKPOINT_DIR=$2
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

export CUCHEM_UI_START_CMD="python3 ./cuchem/cuchem/benchmark/megamolbart.py -o /workspace/megamolbart/benchmark/${ID} -u ${IP_MEGAMOLBART}:50051"
export MEGAMOLBART_CMD="bash -c 'CUDA_VISIBLE_DEVICES=${GPU_ID} python3 launch.py -c ${CHECKPOINT_DIR}'"
export CUCHEM_PATH=/workspace
export MEGAMOLBART_PATH=/workspace/megamolbart

docker-compose \
  --env-file ../../../.env  \
  -f ../../../setup/docker_compose.yml \
  --project-directory ../../../ \
  --project-name "megamolbart${RUN_ID}" \
  ${ACTION}
