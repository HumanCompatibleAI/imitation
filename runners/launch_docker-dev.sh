#!/bin/bash

DOCKER_IMAGE="humancompatibleai/imitation:python-req"
if [[ ${LOCAL_MNT} == "" ]]; then
  LOCAL_MNT="${HOME}/reward-function-transfer"
fi

if [[ ${OUTPUT_MNT} == "" ]]; then
  OUTPUT_MNT="${HOME}/mnt/reward_transfer"
fi

CMD="pip install -e . "
CMD="${CMD} && pip install -e /imitation "
CMD="${CMD} && pip install -e /seals "
CMD="${CMD} && pip install -e /dmc2gym "

# if port is changed here, it should also be changed in scripts/launch_jupyter.sh
FLAGS="-p 9998:9998" 

# Using jupyter lab for easy development
if [[ $1 == "jupyter" ]]; then
  CMD="${CMD} && scripts/launch_jupyter.sh "
fi

docker pull ${DOCKER_IMAGE}
docker run -it --rm --init \
       -v ${HOME}/imitation:/imitation \
       -v ${HOME}/seals:/seals \
       -v ${HOME}/dmc2gym:/dmc2gym \
       -v ${HOME}/.netrc:/root/.netrc \
       -v ${LOCAL_MNT}:/reward-function-transfer \
       -v ${LOCAL_MNT}/mjkey.txt:/root/.mujoco/mjkey.txt \
       -v ${OUTPUT_MNT}:/mnt \
       --env MUJOCO_GL="egl" \
       --env EVAL_OUTPUT_ROOT=/mnt/output \
       ${FLAGS} ${DOCKER_IMAGE} \
       /bin/bash -c "${CMD} && bash"
