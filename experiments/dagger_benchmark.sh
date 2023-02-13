#!/usr/bin/env bash
set -e
source experiments/common.sh

ENVS=(seals_cartpole)
SEEDS=(0 1 2 3 4)
# To prevent race conditions, we use a different log root for each process id.
LOG_ROOT="output/dagger_benchmark/${TIMESTAMP}-${BASHPID}"
extra_configs=()
extra_options=()
extra_parallel_options=()

if ! TEMP=$($GNU_GETOPT -o fTw -l fast,wandb,paper,tmux,pdb,echo,run_name:,log_root:,file_storage: -- "$@"); then
  exit 1
fi
eval set -- "$TEMP"

while true; do
  case "$1" in
    # Fast mode (debug)
    -f | --fast)
      SEEDS=(0)
      extra_configs=("${extra_configs[@]}" environment.fast demonstrations.fast policy_evaluation.fast fast)
      shift
      ;;
    --paper)  # Table benchmark settings
      ENVS=(seals_cartpole seals_mountain_car seals_half_cheetah)
      shift
      ;;
    -w | --wandb)
      # activate wandb logging by adding 'wandb' format string to logging.log_format_strs
      extra_configs=("${extra_configs[@]}" "logging.wandb_logging")
      shift
      ;;
    -T | --tmux)
      extra_parallel_options=("${extra_parallel_options[@]}" --tmux)
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
  python -m imitation.scripts.train_imitation \
  --capture=sys \
  "${extra_options[@]}" \
  dagger \
  with \
  '{env_config_name}' \
  logging.log_dir="${LOG_ROOT}/{env_config_name}_{seed}" \
  dagger.expert_policy_type='ppo' \
  seed='{seed}' \
  "${extra_configs[@]}" \
  ::: env_config_name "${ENVS[@]}" \
  ::: seed "${SEEDS[@]}"

# Directory path is really long. Enter the directory to shorten results output,
# which includes directory of each stdout file.
pushd "${LOG_ROOT}/parallel"
find . -name stderr -print0 | sort -z | xargs -0 tail -n 15 | grep -E '==|Result'
popd

# shellcheck disable=SC2016
echo 'Generate results table using `python -m imitation.scripts.analyze`'
