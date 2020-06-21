#!/usr/bin/env bash
set -e

# This script finds the the mean and standard error of episode return after
# training GAIL or AIRL on benchmark tasks.
#
# The benchmark tasks are defined in the CSV config file
# `experiments/imit_benchmark_config.csv`.

CONFIG_CSV="experiments/imit_benchmark_config.csv"
EXPERT_MODELS_DIR="data/expert_models"
TIMESTAMP=$(date --iso-8601=seconds)
LOG_ROOT="output/imit_benchmark/${TIMESTAMP}"
extra_configs=""
extra_options=""
ALGORITHM="gail"

SEEDS="0 1 2"

TEMP=$(getopt -o f -l fast,gail,airl,run_name:,log_root:,file_storage: -- $@)
if [[ $? != 0 ]]; then exit 1; fi
eval set -- "$TEMP"

while true; do
  case "$1" in
    # Fast mode (debug)
    -f | --fast)
      CONFIG_CSV="tests/data/imit_benchmark_config.csv"
      EXPERT_MODELS_DIR="tests/data/expert_models"
      SEEDS="0"
      extra_configs+="fast "
      shift
      ;;
    --gail)
      ALGORITHM="gail"
      shift
      ;;
    --airl)
      ALGORITHM="airl"
      shift
      ;;
    --run_name)
      extra_options+="--name $2 "
      shift 2
      ;;
    --log_root)
      LOG_ROOT="$2"
      shift 2
      ;;
    --file_storage)
      # Used by `tests/generate_test_data.sh` to save Sacred logs in tests/data.
      extra_options+="--file_storage $2 "
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unrecognized flag $1" >&2
      exit 1
      ;;
  esac
done

mkdir -p "${LOG_ROOT}"
echo "Logging to: ${LOG_ROOT}"

parallel -j 25% --header : --results ${LOG_ROOT}/parallel/ --colsep , --progress \
  python -m imitation.scripts.train_adversarial \
  --capture=sys \
  ${extra_options} \
  with \
  ${ALGORITHM} \
  ${extra_configs} \
  {env_config_name} \
  log_dir="${LOG_ROOT}/{env_config_name}_{seed}/n_expert_demos_{n_expert_demos}" \
  gen_batch_size={gen_batch_size} \
  rollout_path=${EXPERT_MODELS_DIR}/{env_config_name}_0/rollouts/final.pkl \
  checkpoint_interval=0 \
  n_expert_demos={n_expert_demos} \
  seed={seed} \
  :::: $CONFIG_CSV \
  ::: seed ${SEEDS}

# Directory path is really long. Enter the directory to shorten results output,
# which includes directory of each stdout file.
pushd ${LOG_ROOT}/parallel
find . -name stdout | sort | xargs tail -n 15 | grep -E '==|\[result\]'
popd

echo "[Optional] Upload new reward models to S3 (replacing old ones) using the commands:"
echo "aws s3 rm --recursive s3://shwang-chai/public/data/reward_models/${ALGORITHM}/"
echo "aws s3 sync --exclude '*/rollouts/*' --exclude '*/checkpoints/*' --include '*/checkpoints/final/*' '${LOG_ROOT}' s3://shwang-chai/public/data/reward_models/${ALGORITHM}/"

echo 'Generate results table using `python -m imitation.scripts.analyze`'
