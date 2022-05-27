#!/usr/bin/env bash
set -e
SCRIPT_LOC=$(dirname "$0")

ACTION="up -d --scale megamolbart=4"
MODEL_DIR="/models"
CONFIG_DIR="/workspace/benchmark/cuchembench/config"

SOURCE_ROOT=.

while [[ $# -gt 0 ]]; do
  key="$1"
  case $key in
  --stop | -s | stop)
    ACTION=stop
    shift
    ;;
  --config-dir | -c)
    CONFIG_DIR=$2
    shift
    shift
    ;;
  *)
    shift
    ;;
  esac
done

source ${SOURCE_ROOT}/.env

echo ${PYTHONPATH}
python3 -m cuchembench --config-dir ${CONFIG_DIR}
