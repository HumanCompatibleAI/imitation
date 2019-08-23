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

TIMESTAMP=$(${DATE_CMD} --iso-8601=seconds)

OUTPUT_DIR="output/gail_benchmark/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
echo "Logging to: ${OUTPUT_DIR}"

SEEDS="0 1 2"

parallel -j 25% --header : --results ${OUTPUT_DIR}/parallel/ --colsep , --progress \
  python -m imitation.scripts.train_adversarial \
  with \
  "$@" \
  gail \
  {env_config} \
  log_root="${OUTPUT_DIR}" \
  n_gen_steps_per_epoch={n_gen_steps_per_epoch} \
  init_trainer_kwargs.rollout_glob=expert_models/rollouts/{rollout_glob} \
  init_trainer_kwargs.n_expert_demos={n_expert_demos} \
  seed={seed} \
  :::: experiments/gail_benchmark_config.csv \
  ::: seed ${SEEDS}


# Directory path is really long. Enter the directory to shorten results output,
# which includes directory of each stdout file.
pushd ${OUTPUT_DIR}/parallel
find . -name stdout | sort | xargs tail -n 15 | grep -E '==|\[result\]'
popd
