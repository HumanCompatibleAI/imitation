#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.

__usage="launch_docker-dev.sh - Launching humancompatibleai/imitation:python-req

Usage: launch_docker-dev.sh [options] 

options:
  -p, --pull                pull the image to DockerHub

Note: You can specify LOCAL_MNT and MJKEY_MNT environment variables to mount
  local repository and MuJoCo license key respectively.
"

PULL=0

while test $# -gt 0; do
  case "$1" in
  -p | --pull)
    PULL=1 # Pull the image from Docker Hub
    shift
    ;;
  -h | --help)
    echo "${__usage}"
    exit 0
    ;;
  *)
    echo "Unrecognized flag $1" >&2
    exit 1
    ;;
  esac
done

DOCKER_IMAGE="humancompatibleai/imitation:python-req"
# Specify LOCAL_MNT if you want to mount a local directory to the docker container
if [[ ${LOCAL_MNT} == "" ]]; then
  LOCAL_MNT="${HOME}/imitation"
fi

# Pass your own mjkey.txt
if [[ ${MJKEY_MNT} == "" ]]; then
  MJKEY_MNT="${HOME}/mnt/mjkey.txt"
fi

# install imitation in developer mode
CMD="pip install -e .[docs,parallel,test] gym[mujoco]" # borrowed from ci/build_and_activate_venv.sh

# Pull image from DockerHub if prompted
if [[ $PULL == 1 ]]; then
  echo "Pulling ${DOCKER_IMAGE} from DockerHub"
  docker pull ${DOCKER_IMAGE}
fi

docker run -it --rm --init \
  -v "${LOCAL_MNT}:/imitation" \
  -v "${MJKEY_MNT}:/root/.mujoco/mjkey.txt" \
  ${DOCKER_IMAGE} \
  /bin/bash -c "${CMD} && exec bash"
