#!/usr/bin/env bash
# This script regenerates tests/data.
set -e

expert_models_dir=tests/data/expert_models
reward_models_dir=tests/data/reward_models
SAVE_DIRS=($expert_models_dir $reward_models_dir)

# Wipe directories for writing later.
for save_dir in ${SAVE_DIRS}; do
  if [[ -d ${save_dir} ]]; then
    rm -r ${save_dir}
  fi
  mkdir -p ${save_dir}
done


# Regenerate tests/data/expert_models (for various tests).
experiments/train_experts.sh -r


# Regenerate tests/data/reward_models (for `experiments/transfer_learn_benchmark.sh -f`).
experiments/imit_benchmark.sh -f --gail --log_root "${reward_models_dir}/gail"
