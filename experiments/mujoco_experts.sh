#!/usr/bin/env bash

ENVS="acrobot cartpole mountain_car reacher half_cheetah hopper ant humanoid swimmer walker"
SEEDS="0 1 2"

if $(command -v gdate > /dev/null); then
  DATE_CMD=gdate  # macOS compatibility
else
  DATE_CMD=date
fi

TIMESTAMP=$(${DATE_CMD} --iso-8601=seconds)
OUTPUT_DIR=output/mujoco_experts/${TIMESTAMP}/

parallel -j 25% --header : --results ${OUTPUT_DIR}/parallel/ \
         python -m imitation.scripts.data_collect with log_root=${OUTPUT_DIR} {env} seed={seed} \
         ::: env ${ENVS} ::: seed ${SEEDS}
