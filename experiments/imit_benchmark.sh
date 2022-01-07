#!/usr/bin/env bash
set -e

# This script finds the the mean and standard error of episode return after
# training GAIL or AIRL on benchmark tasks.
#
# The benchmark tasks are defined in the CSV config file
# `experiments/imit_benchmark_config.csv`.

source experiments/common.sh

SEEDS=(0 1 2 3 4)
CONFIG_CSV="experiments/imit_benchmark_config.csv"
DATA_DIR="${DATA_DIR:-data/}"
LOG_ROOT="output/imit_benchmark/${TIMESTAMP}"
extra_configs=()
extra_options=()
extra_parallel_options=()
ALGORITHM="gail"

if ! TEMP=$($GNU_GETOPT -o f,T,w -l fast,gail,airl,mvp_seals,cheetah,tmux,pdb,run_name:,log_root:,file_storage:,echo,wandb -- "$@"); then
  exit 1
fi
eval set -- "$TEMP"

while true; do
  case "$1" in
    # Fast mode (debug)
    -f | --fast)
      CONFIG_CSV="tests/testdata/imit_benchmark_config.csv"
      DATA_DIR="tests/testdata/"
      SEEDS=(0)
      extra_configs=("${extra_configs[@]}" common.fast demonstrations.fast rl.fast train.fast fast)
      shift
      ;;
    -w | --wandb)
      # activate wandb logging by adding 'wandb' format string to common.log_format_strs
      extra_configs=("${extra_configs[@]}" "common.wandb_logging")
      shift
      ;;
    --mvp_seals)
      CONFIG_CSV="experiments/imit_table_mvp_seals_config.csv"
      shift
      ;;
    --cheetah)
      CONFIG_CSV="experiments/imit_table_cheetahs.csv"
      shift
      ;;
    -T | --tmux)
      extra_parallel_options=("${extra_parallel_options[@]}" --tmux)
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
      extra_options=("${extra_options[@]}" --name "$2")
      shift 2
      ;;
    --log_root)
      LOG_ROOT="$2"
      shift 2
      ;;
    --file_storage)
      # Used by `tests/generate_test_data.sh` to save Sacred logs in tests/testdata.
      extra_options=("${extra_options[@]}" --file_storage "$2")
      shift 2
      ;;
    --pdb)
      # shellcheck disable=SC2016
      echo 'NOTE: Interact with PDB session via tmux. If an error occurs, `parallel` '
      echo 'will hang and wait for user input in tmux session.'
      # Needed for terminal output.
      extra_parallel_options=("${extra_parallel_options[@]}" --tmux)
      extra_options=("${extra_options[@]}" --pdb)
      shift
      ;;
    --echo)
      extra_parallel_options=("${extra_parallel_options[@]}" echo)
      shift
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

parallel -j 25% --header : --results "${LOG_ROOT}/parallel/" --colsep , --progress \
  "${extra_parallel_options[@]}" \
  python -m imitation.scripts.train_adversarial \
  --capture=sys \
  "${extra_options[@]}" \
  "${ALGORITHM}" \
  with \
  '{env_config_name}' seed='{seed}' \
  common.log_dir="${LOG_ROOT}/{env_config_name}_{seed}/n_expert_demos_{n_expert_demos}" \
  demonstrations.data_dir="${DATA_DIR}" \
  demonstrations.n_expert_demos='{n_expert_demos}' \
  checkpoint_interval=0 \
  "${extra_configs[@]}" \
  :::: $CONFIG_CSV \
  ::: seed "${SEEDS[@]}"

# Directory path is really long. Enter the directory to shorten results output,
# which includes directory of each stdout file.
pushd "${LOG_ROOT}/parallel"
find . -name stdout -print0 | sort -z | xargs -0 tail -n 15 | grep -E '==|Result'
popd

echo "[Optional] Upload new reward models to S3 (replacing old ones) using the commands:"
echo "aws s3 rm --recursive s3://shwang-chai/public/data/reward_models/${ALGORITHM}/"
echo "aws s3 sync --exclude '*/rollouts/*' --exclude '*/checkpoints/*' --include '*/checkpoints/final/*' '${LOG_ROOT}' s3://shwang-chai/public/data/reward_models/${ALGORITHM}/"

# shellcheck disable=SC2016
echo 'Generate results table using `python -m imitation.scripts.analyze`'
