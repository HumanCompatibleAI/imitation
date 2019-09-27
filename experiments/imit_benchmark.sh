#!/usr/bin/env bash

# This script finds the the mean and standard error of episode return after
# training GAIL or AIRL on benchmark tasks.
#
# The benchmark tasks are defined in the CSV config file
# `experiments/imit_benchmark_config.csv`.

if $(command -v gdate > /dev/null); then
  DATE_CMD=gdate  # macOS compatibility
else
  DATE_CMD=date
fi

RUN_NAME=${RUN_NAME:-no_run_name}
USE_GAIL=${USE_GAIL:-True}
CONFIG_CSV="experiments/imit_benchmark_config.csv"
EXPERT_MODELS_DIR="expert_models"
TIMESTAMP=$(${DATE_CMD} --iso-8601=seconds)
LOG_DIR="output/imit_benchmark/${TIMESTAMP}"
mkdir -p "${OUTPUT_DIR}"
echo "Logging to: ${OUTPUT_DIR}"

SEEDS="0 1 2"

getopt -o f -l fast,gail,airl,run_name:,log_dir:
if [[ $? != 0 ]]; exit 1; fi
eval set -- "$TEMP"

while true; do
  case "$1" in
    # Fast mode (debug)
    -f | --fast)
      CONFIG_CSV="tests/data/imit_benchmark_config.csv"
      EXPERT_MODELS_DIR="tests/data"
      SEEDS="0"
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
    --log_dir)
      LOG_DIR="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Parsing error" > &2
      exit 1
      ;;
  esac
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
  init_trainer_kwargs.use_gail=${USE_GAIL} \
  seed={seed} \
  :::: $CONFIG_CSV \
  ::: seed ${SEEDS}

# Directory path is really long. Enter the directory to shorten results output,
# which includes directory of each stdout file.
pushd ${OUTPUT_DIR}/parallel
find . -name stdout | sort | xargs tail -n 15 | grep -E '==|\[result\]'
popd
