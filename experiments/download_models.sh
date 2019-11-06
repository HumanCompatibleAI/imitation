#!/bin/bash

# Always sync to ../data, relative to this script.
SCRIPT_DIR="$(cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd)"
PROJECT_DIR="$(dirname ${SCRIPT_DIR})"
DATA_DIR="${PROJECT_DIR}/data"

if [[ -d ${DATA_DIR} ]]; then
  rm -r ${DATA_DIR}
fi
aws s3 sync --exclude '*/rollouts/*' --exclude '*/checkpoints/*' --exclude '*/policies/*' --include "*/checkpoints/final/*" --include "*/policies/final/*" s3://shwang-chai/public/data/ ${DATA_DIR}
