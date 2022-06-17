#!/usr/bin/env bash

set -e  # exit immediately on any error

venv=$1
if [[ ${venv} == "" ]]; then
	venv="venv"
fi

virtualenv -p python3.8 ${venv}
# shellcheck disable=SC1090,SC1091
source ${venv}/bin/activate
python -m pip install --upgrade pip
pip install ".[docs,parallel,test,mujoco]"
