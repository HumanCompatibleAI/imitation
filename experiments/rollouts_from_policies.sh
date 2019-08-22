#!/usr/bin/env bash

# This script loads expert PPO2 policies of the form
# `${EXPERT_MODELS_DIR}/policies/{env_name}/{uuid}/`
# and generates rollouts. The rollouts are saved to
# `${EXPERT_MODELS_DIR}/rollouts/{env_name}/{uuid}.pkl`.
#
# The values of {env_name} are defined in the config file
# `experiments/rollouts_from_policies.csv`.
# {uuid} is dynamically generated to match every policy directory. In other
# words, we generate rollouts for every policy
# `${EXPERT_MODELS_DIR}/policies/{env_name}/*/`.
#
# EXPERT_MODELS_DIR is "./expert_models" by default, but can be configured
# via `export EXPERT_MODELS_DIR=foobar`.

if $(command -v gdate > /dev/null); then
  DATE_CMD=gdate  # macOS compatibility
else
  DATE_CMD=date
fi

if $(command -v gfind > /dev/null); then
  # macOS compatibility. Install `gfind` with `brew install find-utils`.
  # The macOS version of find doesn't support the `-printf` option.
  FIND_CMD=gfind
else
  FIND_CMD=find
fi

TIMESTAMP=$(${DATE_CMD} --iso-8601=seconds)
EXPERT_MODELS_DIR=${EXPERT_MODELS_DIR:-./expert_models}
OUTPUT_DIR_PREFIX=${OUTPUT_DIR_PREFIX:-.}
CONFIG_CSV=${CONFIG_CSV:-experiments/rollouts_from_policies.csv}
config_csv_abs=$(realpath ${CONFIG_CSV})

OUTPUT_DIR="${OUTPUT_DIR_PREFIX}/output/train_experts/${TIMESTAMP}"
output_dir_abs="$(realpath ${OUTPUT_DIR})"

echo "Loading config from ${CONFIG_CSV}"
echo "Loading expert models from ${EXPERT_MODELS_DIR}"
echo "Writing logs in ${OUTPUT_DIR}"

pushd ${EXPERT_MODELS_DIR} > /dev/null

# Explanation:
#
# This is a nested parallel command.
#
# The outer `parallel` first loads the replacement variables {env_name},
# {config_name}, and {n_demonstrations} from the CSV file ${config_csv_abs}.
# Then it executes `find` to to dynamically generate every {uuid}
# associated with {env_name}.
#
# The inner parallel reads in values of {uuid} from STDIN and executes
# `python -m imitation.scripts.expert_demos` with the appropriate Sacred options.

parallel -j 25% --header : --results ${output_dir_abs}/parallel/ --colsep , \
  "${FIND_CMD} policies -mindepth 2 -maxdepth 2 -type d -path '*{env_name}/*' \
    -printf '%f\n' \
  | parallel -I '{uuid}' -j 1 \
    python -m imitation.scripts.expert_demos rollouts_from_policy \
    with \
    {config_name} \
    log_root='${output_dir_abs}' \
    policy_path='policies/{env_name}/{uuid}' \
    rollout_save_dir='rollouts/{env_name}/{uuid}' \
    rollout_save_n_episodes='{n_demonstrations}' " \
  :::: ${config_csv_abs}

popd > /dev/null
