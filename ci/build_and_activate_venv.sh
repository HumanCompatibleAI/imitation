#!/usr/bin/env bash
#
# Usage: ./build_and_activate_venv.sh [venv_path] [python_version]
#   venv_path: Path at which the virtualenv directory should be created. Defaults to
#     'venv'.
#   python_version: Version of python to be used in the virtualenv. Defaults to
#     'python3.8'.

set -e  # exit immediately on any error

venv=$1
if [[ ${venv} == "" ]]; then
  venv="venv"
fi
python_version=$2
if [[ ${python_version} == "" ]]; then
  python_version="python3.8"
fi

virtualenv -p ${python_version} ${venv}
# shellcheck disable=SC1090,SC1091
source ${venv}/bin/activate

# If platform is linux, install pytorch CPU version.
# This will prevent installing the CUDA version in the pip install ".[docs,parallel,test]" command.
# The CUDA version is a couple of gigabytes larger than the CPU version.
# Since we don't need the CUDA version for testing, we can save some time by not installing it.
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
  pip install torch --index-url https://download.pytorch.org/whl/cpu
fi
pip install ".[docs,parallel,test]"
