#!/usr/bin/env bash

ENVS+="acrobot cartpole mountain_car "
ENVS+="reacher half_cheetah hopper ant humanoid swimmer walker "
ENVS+="two_d_maze custom_ant disabled_ant "
SEEDS="0 1 2"

if $(command -v gdate > /dev/null); then
  DATE_CMD=gdate  # macOS compatibility
else
  DATE_CMD=date
fi

TIMESTAMP=$(${DATE_CMD} --iso-8601=seconds)
OUTPUT_DIR=output/train_experts/${TIMESTAMP}/

echo "Writing logs in ${OUTPUT_DIR}"

parallel -j 25% --header : --progress --results ${OUTPUT_DIR}/parallel/ \
  python -m imitation.scripts.data_collect \
  with \
  {env} \
  seed={seed} \
  log_root=${OUTPUT_DIR} \
  ::: env ${ENVS} ::: seed ${SEEDS}
