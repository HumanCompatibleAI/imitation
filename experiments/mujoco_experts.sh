#!/usr/bin/env bash

ENVS="acrobot cartpole mountaincar reacher halfcheetah hopper ant humanoid"
SEEDS="0 1 2"

TIMESTAMP=$(date --iso-8601=seconds)
OUTPUT_DIR=outputs/mujoco_experts/${TIMESTAMP}/

parallel -j 25% --header : --results ${OUTPUT_DIR}/parallel/ \
         python -m imitation.scripts.data_collect with log_root=${OUTPUT_DIR} {env} seed={seed} \
         ::: env ${ENVS} ::: seed ${SEEDS}
