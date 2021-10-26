#!/usr/bin/env bash
set -e
SCRIPT_LOC=$(dirname "$0")

ID=100
ACTION="up"
GPU_ID="0"
MODEL_DIR="/models"
CONFIG_DIR="/workspace/cuchem/benchmark/scripts"
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
source ${SCRIPT_LOC}/../../../.env
export RUN_ID="_${ID}"
export PLOTLY_PORT="5${ID}"

export SUBNET=192.${ID}.100.0/16
export IP_CUCHEM_UI=192.${ID}.100.1
export IP_MEGAMOLBART=192.${ID}.100.2

export CUCHEM_UI_START_CMD="python3 ./cuchem/cuchem/benchmark/megamolbart.py \
  --config-dir ${CONFIG_DIR}"

export MEGAMOLBART_CMD="bash -c 'CUDA_VISIBLE_DEVICES=${GPU_ID} python3 launch.py'"

export CUCHEM_PATH=/workspace
export MEGAMOLBART_PATH=/workspace/megamolbart
export WORKSPACE_DIR="$(pwd)"

# Create ExCAPE database directory
EXCAPE_DIR=${CONTENT_PATH}/data/ExCAPE
mkdir -p ${EXCAPE_DIR}
set -x
if [ ! -e ${EXCAPE_DIR}/pubchem.chembl.dataset4publication_inchi_smiles_v2.tsv ]
then
  if [ ! -e ${EXCAPE_DIR}/pubchem.chembl.dataset4publication_inchi_smiles_v2.tsv.xz ]
  then
    # Download
    wget https://zenodo.org/record/2543724/files/pubchem.chembl.dataset4publication_inchi_smiles_v2.tsv.xz?download=1 \
      -O ${EXCAPE_DIR}/pubchem.chembl.dataset4publication_inchi_smiles_v2.tsv.xz
  fi
  cd ${EXCAPE_DIR} && zstd -d pubchem.chembl.dataset4publication_inchi_smiles_v2.tsv.xz
fi

echo $ID
echo $SUBNET
docker-compose \
  -f ${SCRIPT_LOC}/../../../setup/docker_compose.yml \
  --project-directory ${SCRIPT_LOC}/../../../ \
  --project-name "megamolbart${RUN_ID}" \
  ${ACTION}
