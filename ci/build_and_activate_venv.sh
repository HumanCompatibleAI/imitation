#!/usr/bin/env bash

set -e  # exit immediately on any error

program_name=$0

usage() {
  echo "usage: ${program_name} [--skip-mujoco] [venv_path]";
  echo "  --skip-mujoco: Skip installing MuJoCo-related packages."
  echo "  venv_path: Path at which the virtualenv directory should be created. Defaults"
  echo "    to 'venv'."
}

# Transform long options to short ones, as getopts doesn't support long options
# (https://stackoverflow.com/a/30026641/4865149)
for arg in "$@"; do
  shift
  case "$arg" in
    '--skip-mujoco') set -- "$@" '-m'   ;;
    *)               set -- "$@" "$arg" ;;
  esac
done

skip_mujoco=false
# Read command line flags
while getopts ":m" opt; do
  case $opt in
    m)
      skip_mujoco=true
      echo "--skip-mujoco was triggered!" >&2
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      usage
      exit 1
      ;;
  esac
done
# shift all processed options away so that positional arguments can be accessed with
# $1, $2, etc.
shift $((OPTIND-1))

venv=$1
if [[ ${venv} == "" ]]; then
  venv="venv"
fi

virtualenv -p python3.8 ${venv}
# shellcheck disable=SC1090,SC1091
source ${venv}/bin/activate
python -m pip install --upgrade pip
if [[ ${skip_mujoco} ]]; then
  pip install ".[docs,parallel,test]"
else
  pip install ".[docs,parallel,test,mujoco]"
fi
