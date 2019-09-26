#!/usr/bin/env bash

# This script finds the the mean and standard error of episode return after
# training GAIL on benchmark tasks.
#
# The benchmark tasks are defined in the CSV config file
# `experiments/gail_benchmark_config.csv`.
#
# The CSV configuration can be modified and regenerated from following spreadsheet:
# https://docs.google.com/spreadsheets/d/1MZYTr23ddwhO2PrNI2EalKleoULXlsH5y9oqawt22pY/edit?usp=sharing

if $(command -v gdate > /dev/null); then
  DATE_CMD=gdate  # macOS compatibility
else
  DATE_CMD=date
fi

CONFIG_CSV="experiments/gail_benchmark_config.csv"
EXPERT_MODELS_DIR="expert_models"
TIMESTAMP=$(${DATE_CMD} --iso-8601=seconds)
OUTPUT_DIR="output/gail_benchmark/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
echo "Logging to: ${OUTPUT_DIR}"

SEEDS="0 1 2"

# Fast mode (debug)
while getopts "f" arg; do
  if [[ $arg == "f" ]]; then
    CONFIG_CSV="tests/data/gail_benchmark_config.csv"
    EXPERT_MODELS_DIR="tests/data"
    SEEDS="0"
  fi
done

parallel -j 25% --header : --results ${OUTPUT_DIR}/parallel/ --colsep , --progress \
  python -m imitation.scripts.train_adversarial \
  with \
  "$@" \
  gail \
  {env_config_name} \
  log_root="${OUTPUT_DIR}" \
  n_gen_steps_per_epoch={n_gen_steps_per_epoch} \
  rollout_path=${EXPERT_MODELS_DIR}/{env_config_name}_0/rollouts/auto.pkl \
  n_expert_demos={n_expert_demos} \
  seed={seed} \
  :::: $CONFIG_CSV \
  ::: seed ${SEEDS}

# Directory path is really long. Enter the directory to shorten results output,
# which includes directory of each stdout file.
pushd ${OUTPUT_DIR}/parallel
find . -name stdout | sort | xargs tail -n 15 | grep -E '==|\[result\]'
popd
