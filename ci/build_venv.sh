#!/usr/bin/env bash

set -e  # exit immediately on any error

venv=$1
if [[ ${venv} == "" ]]; then
	venv="venv"
fi

virtualenv -p python3.7 ${venv}
source ${venv}/bin/activate
pip install .[cpu,docs,jax,parallel,test] gym[mujoco]
