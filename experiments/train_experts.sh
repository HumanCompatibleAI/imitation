#!/usr/bin/env bash

# This script trains experts for AIRL and GAIL benchmark scripts.
# When training is finished, it reports the mean episode reward of each
# expert and builds a zip file of expert models that can be used for
# `experiments/gail_benchmark.sh`.

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
OUTPUT_DIR="output/train_experts/${TIMESTAMP}"
RESULTS_FILE="results.txt"
extra_configs=""

while getopts "fr" arg; do
  if [[ $arg == "f" ]]; then
    # Fast mode (debug)
    ENVS="cartpole pendulum"
    SEEDS="0"
    extra_configs="fast"
  elif [[ $arg == "r" ]]; then
    # Regenerate test data (policies and rollouts).
    #
    # Combine with fast mode flag to generate low-computation versions of
    # test data.
    # Use `git clean -df tests/data` to remove extra log files.
    ENVS="cartpole pendulum"
    SEEDS="0"
    OUTPUT_DIR="tests/data"
    extra_configs="rollout_save_n_episodes=50 rollout_save_n_timesteps=None"
  fi
done

echo "Writing logs in ${OUTPUT_DIR}"
# Train experts.
parallel -j 25% --header : --progress --results ${OUTPUT_DIR}/parallel/ \
  python -m imitation.scripts.expert_demos \
  with \
  {env} ${extra_configs} \
  seed={seed} \
  log_dir="${OUTPUT_DIR}/{env}_{seed}" \
  ::: env ${ENVS} ::: seed ${SEEDS}

pushd $OUTPUT_DIR

# Display and save mean episode reward to ${RESULTS_FILE}.
find . -name stdout | xargs tail -n 15 | grep -E '(==|ep_reward_mean)' | tee ${RESULTS_FILE}

# Build zipfile using the directory structure corresponding to the symlinks
# from before.
ZIP_FILE="expert_models.zip"
zip --exclude 'rollouts/*' -rv ${ZIP_FILE} ${RESULTS_FILE} */

echo "Expert zip saved to $(pwd)/${ZIP_FILE}"

popd
