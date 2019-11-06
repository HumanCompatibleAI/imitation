#!/bin/bash

# Always sync to ../data, relative to this script.
SCRIPT_DIR="$(cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd)"
PROJECT_DIR="$(dirname ${SCRIPT_DIR})"

aws s3 sync s3://shwang-chai/public/data/ ${PROJECT_DIR}/data
