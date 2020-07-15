#!/usr/bin/env bash

# This script loads expert PPO policies of the form
# `${EXPERT_MODELS_DIR}/{env_config_name}_0/policies/final/`
# and generates rollouts. The rollouts are saved to
# `${EXPERT_MODELS_DIR}/{env_config_name}_0/rollouts/final.pkl`
#
# EXPERT_MODELS_DIR is "data/expert_models" by default, but can be configured
# via `export EXPERT_MODELS_DIR=foobar`.
#
# The values of {env_config_name} are defined in the config file
# `experiments/rollouts_from_policies.csv`.
#
# TODO(shwang): When we are migrated to Python, first evaluate the
# mean return of each policy, then use this to choose the best seed rather
# than hardcoding seed 0.

TIMESTAMP=$(date --iso-8601=seconds)
EXPERT_MODELS_DIR=${EXPERT_MODELS_DIR:-data/expert_models}
CONFIG_CSV=${CONFIG_CSV:-experiments/rollouts_from_policies_config.csv}
OUTPUT_DIR="output/train_experts/${TIMESTAMP}"

while getopts "f" arg; do
  if [[ $arg == "f" ]]; then
    # Fast mode (debug)
    CONFIG_CSV="tests/data/rollouts_from_policies_config.csv"
    EXPERT_MODELS_DIR="tests/data/expert_models"
  fi
done

echo "Loading config from ${CONFIG_CSV}"
echo "Loading expert models from ${EXPERT_MODELS_DIR}"
echo "Writing logs in ${OUTPUT_DIR}"

parallel -j 25% --header : --results ${OUTPUT_DIR}/parallel/ --colsep , \
  python -m imitation.scripts.expert_demos rollouts_from_policy \
  --capture=sys \
  with \
  {env_config_name} \
  log_root="${OUTPUT_DIR}" \
  policy_path="${EXPERT_MODELS_DIR}/{env_config_name}_0/policies/final/" \
  rollout_save_path="${EXPERT_MODELS_DIR}/{env_config_name}_0/rollouts/final.pkl" \
  rollout_save_n_episodes="{n_demonstrations}" \
  rollout_save_n_timesteps=None \
  :::: ${CONFIG_CSV}
