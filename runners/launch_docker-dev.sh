#!/bin/bash

set -e # Exit immediately if a command exits with a non-zero status.
set -x # echo on

__usage="launch_docker-dev.sh - Launching humancompatibleai/imitation:python-req

Usage: launch_docker-dev.sh [options] 

options:
  -p, --pull                pull the image to DockerHub
  -h, --help                show this help message and exit
  -r, --remove_custom       remove customized config mounts for tmux and wandb

Note: You can specify IMIT_LOCAL_MNT environment variables to mount local
  repository and MuJoCo license key respectively.
"

PULL=0
REMOVE_CUSTOM=0

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

CUSTOM_HOME="/nas/ucb/yawen"

DOCKER_IMAGE="humancompatibleai/imitation:python-req"
# Specify IMIT_LOCAL_MNT if you want to mount a local directory to the docker container
if [[ ${IMIT_LOCAL_MNT} == "" ]]; then
  IMIT_LOCAL_MNT="${CUSTOM_HOME}/imitation"
fi

###########
## Flags ##
###########

# if port is changed here, it should also be changed in scripts/launch_jupyter.sh
FLAGS+="-p 9988:9988 "  # ports

if [[ $REMOVE_CUSTOM == 0 ]]; then
  # flags for easy development
  FLAGS+="-v ${CUSTOM_HOME}/custom_configs/.netrc:/root/.netrc "  # mounting local .netrc for wandb login
  FLAGS+="-v ${CUSTOM_HOME}/custom_configs/.tmux.conf:/root/.tmux.conf "  # mounting local .tmux.conf for custom tmux configs
fi


# install imitation in developer mode
CMD+="apt-get update -q && apt-get install -y --no-install-recommends tmux "
CMD+="&& pip install -e .[docs,parallel,test] gym[mujoco] " # copied from ci/build_and_activate_venv.sh

# Pull image from DockerHub if prompted
if [[ $PULL == 1 ]]; then
  echo "Pulling ${DOCKER_IMAGE} from DockerHub"
  docker pull ${DOCKER_IMAGE}
fi


docker run -it --rm --init \
  ${FLAGS} \
  -v "${IMIT_LOCAL_MNT}:/imitation" \
  -v "${CUSTOM_HOME}/mnt/reward_transfer:/mnt" \
  ${DOCKER_IMAGE} \
  /bin/bash -c "${CMD} && exec bash"
