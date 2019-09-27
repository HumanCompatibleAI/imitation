#!/usr/bin/env bash

# This script finds the the mean and standard error of episode return after
# training GAIL or AIRL on benchmark tasks.
#
# The benchmark tasks are defined in the CSV config file
# `experiments/imit_benchmark_config.csv`.

RUN_NAME=${RUN_NAME:-no_run_name}
USE_GAIL=${USE_GAIL:-True}
CONFIG_CSV="experiments/imit_benchmark_config.csv"
EXPERT_MODELS_DIR="expert_models"
TIMESTAMP=$(date --iso-8601=seconds)
LOG_ROOT="output/imit_benchmark/${TIMESTAMP}"
extra_configs=""
extra_options=""
mkdir -p "${LOG_ROOT}"
echo "Logging to: ${LOG_ROOT}"

SEEDS="0 1 2"

TEMP=$(getopt -o f -l fast,gail,airl,run_name:,log_root: -- $@)
if [[ $? != 0 ]]; then exit 1; fi
eval set -- "$TEMP"

while true; do
  case "$1" in
    # Fast mode (debug)
    -f | --fast)
      CONFIG_CSV="tests/data/imit_benchmark_config.csv"
      EXPERT_MODELS_DIR="tests/data"
      SEEDS="0"
      extra_configs+="fast "
      shift
      ;;
    --gail)
      USE_GAIL="True"
      shift
      ;;
    --airl)
      USE_GAIL="False"
      shift
      ;;
    --run_name)
      RUN_NAME="$2"
      shift 2
      ;;
    --log_root)
      LOG_ROOT="$2"
      shift 2
      ;;
    --file_storage)
      extra_options+="--file_storage $2 "
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "$1"
      echo "Parsing error" >&2
      exit 1
      ;;
  esac
done

parallel -j 25% --header : --results ${LOG_ROOT}/parallel/ --colsep , --progress \
  python -m imitation.scripts.train_adversarial \
  --name ${RUN_NAME} \
  ${extra_options} \
  with \
  gail \
  ${extra_configs} \
  {env_config_name} \
  log_root="${LOG_ROOT}" \
  n_gen_steps_per_epoch={n_gen_steps_per_epoch} \
  rollout_path=${EXPERT_MODELS_DIR}/{env_config_name}_0/rollouts/final.pkl \
  n_expert_demos={n_demonstrations} \
  init_trainer_kwargs.use_gail=${USE_GAIL} \
  seed={seed} \
  :::: $CONFIG_CSV \
  ::: seed ${SEEDS}

# Directory path is really long. Enter the directory to shorten results output,
# which includes directory of each stdout file.
pushd ${LOG_ROOT}/parallel
find . -name stdout | sort | xargs tail -n 15 | grep -E '==|\[result\]'
popd
