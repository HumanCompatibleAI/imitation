#!/usr/bin/env bash
# This script regenerates tests/data.
set -e


TEMP=$(getopt -o '' -l all -- $@)
if [[ $? != 0 ]]; then exit 1; fi
eval set -- "$TEMP"

while true; do
  case "$1" in
    --all)
      # If `--all` option is provided then regenerate CartPole and Pendulum
      # PPO2 experts via `train_experts.sh -r`. By default we skip this
      # step because it takes several minutes to generate good experts.
      #
      # We need good (rather than quickly generated dummy) experts because
      # `tests/test_{bc,density_baselines}.py` require particular performance
      # measures.
      ALL_MODE=true
      shift
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unrecognized argument $1"
      exit 1
      ;;
  esac
done


sacred_dir=tests/data/imit_benchmark_sacred
reward_models_dir=tests/data/reward_models
SAVE_DIRS=($sacred_dir $reward_models_dir)
if $ALL_MODE; then
  SAVE_DIRS+=('tests/data/expert_models')
fi

# Wipe directories for writing later.
for save_dir in ${SAVE_DIRS}; do
  if [[ -d ${save_dir} ]]; then
    rm -r ${save_dir}
  fi
  mkdir -p ${save_dir}
done


# Regenerate Cartpole and Pendulum expert rollouts and policies
# (requires --all option).
if $ALL_MODE; then
  experiments/train_experts.sh -r
else
  echo "Skipping CartPole and Pendulum expert generation..."
fi


# Regenerate tests/data/imit_benchmark_sacred (for analysis tests)
# and tests/data/reward_models (for `experiments/transfer_learn_benchmark.sh -f`).
alias imit_benchmark="experiments/imit_benchmark.sh -f --file_storage=${sacred_dir}"
shopt -s expand_aliases

imit_benchmark --gail --run_name BAR --log_root "${reward_models_dir}/gail"
# Only need one reward model, so leave the rest in temporary directories.
# The sacred logs are still saved to ${sacred_dir} and used for analysis test.
imit_benchmark --gail --run_name BAR --log_root "$(mktemp -d)"
imit_benchmark --airl --run_name FOO --log_root "$(mktemp -d)"
