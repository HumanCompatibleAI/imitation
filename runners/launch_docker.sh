#!/bin/bash

DOCKER_IMAGE="kmdanielduan/reward-function-transfer:python-req"
if [[ ${LOCAL_MNT} == "" ]]; then
  LOCAL_MNT="${HOME}/reward-function-transfer"
fi

CMD="python setup.py sdist bdist_wheel && pip install --upgrade --force-reinstall dist/reward_function_transfer-*.whl && bash"
FLAGS=""

docker pull ${DOCKER_IMAGE}

docker run -it --rm \
       -v ${LOCAL_MNT}/data:/mnt/reward_transfer/data \
       -v ${LOCAL_MNT}:/reward-function-transfer \
       -v ${LOCAL_MNT}/mjkey.txt:/root/.mujoco/mjkey.txt \
       --env MUJOCO_GL="egl" \
       --env EVAL_OUTPUT_ROOT=/mnt/reward_transfer/data \
       ${FLAGS} ${DOCKER_IMAGE} \
       /bin/bash -c "${CMD}"
