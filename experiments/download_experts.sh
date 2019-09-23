#!/bin/bash

# Always sync to ../expert_models, relative to this script.
SCRIPT_DIR="$(cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd)"
PROJECT_DIR="$(dirname ${SCRIPT_DIR})"
EXPERT_MODELS_DIR=${PROJECT_DIR}/expert_models

aws s3 sync s3://shwang-chai/public/expert_models/ ${EXPERT_MODELS_DIR}
