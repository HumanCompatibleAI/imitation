#!/usr/bin/env bash
#
# Usage: ./build_and_activate_venv.sh [venv_path] [python_version]
#   venv_path: Path at which the virtualenv directory should be created. Defaults to
#     'venv'.
#   python_version: Version of python to be used in the virtualenv. Defaults to
#     'python3.8'.

set -e  # exit immediately on any error

atari_roms=$1
venv=$2
if [[ ${venv} == "" ]]; then
  venv="venv"
fi
python_version=$3
if [[ ${python_version} == "" ]]; then
  python_version="python3.8"
fi

virtualenv -p ${python_version} ${venv}
# shellcheck disable=SC1090,SC1091
source ${venv}/bin/activate
# Note: We need to install setuptools==66.1.1 to allow installing gym==0.21.0.
python -m pip install --upgrade pip setuptools==66.1.1

# download roms and separately install autorom
pip install autorom
curl "${atari_roms}" -o Roms.tar.gz.b64
base64 Roms.tar.gz.b64 --decode &> Roms.tar.gz
AutoROM --accept-license --source-file Roms.tar.gz

pip install ".[docs,parallel,test]"
