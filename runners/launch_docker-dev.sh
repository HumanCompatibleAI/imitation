#!/bin/bash

# Usage: launch_docker-dev.sh
# You can specify LOCAL_MNT and MJKEY_MNT ahead

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
CMD="pip install -e .[docs,parallel,test] gym[mujoco]"  # borrowed from ci/build_venv.sh

# docker pull ${DOCKER_IMAGE}  # Uncomment this line to pull the image
docker run -it --rm --init \
       -v ${LOCAL_MNT}:/imitation \
       -v ${MJKEY_MNT}:/root/.mujoco/mjkey.txt \
       ${FLAGS} ${DOCKER_IMAGE} \
       /bin/bash -c "${CMD} && bash"
