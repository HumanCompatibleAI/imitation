#!/usr/bin/env bash

set -e  # exit immediately on any error

venv=venv
virtualenv -p python3.7 ${venv}
source ${venv}/bin/activate
pip install .[cpu,dev,test]
