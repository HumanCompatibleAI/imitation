#!/usr/bin/env bash
set -e
source experiments/common.sh

CONFIG_CSV=${CONFIG_CSV:-experiments/rollouts_from_policies_config.csv}

ENVS=(seals_ant seals_half_cheetah)
SEEDS=(0 1 2 3 4)
# To prevent race conditions, we use a different log root for each process id.
LOG_ROOT="output/dagger_benchmark/${TIMESTAMP}-${BASHPID}"
extra_configs=()
extra_options=()
extra_parallel_options=()
print_config=""

if ! TEMP=$($GNU_GETOPT -o fTwp -l fast,wandb,paper,tmux,pdb,echo,run_name:,log_root:,file_storage:,timestamp:,print -- "$@"); then
  exit 1
fi
eval set -- "$TEMP"

while true; do
  case "$1" in
    # Fast mode (debug)
    -f | --fast)
      SEEDS=(0)
      extra_configs=("${extra_configs[@]}" common.fast demonstrations.fast train.fast fast)
      shift
      ;;
    --paper)  # Table benchmark settings
      ENVS=(seals_cartpole seals_mountain_car seals_half_cheetah)
      shift
      ;;
    -w | --wandb)
      # activate wandb logging by adding 'wandb' format string to common.log_format_strs
      extra_configs=("${extra_configs[@]}" "common.wandb_logging" "common.wandb.wandb_kwargs.project='algorithm-benchmark' ")
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
    --timestamp)
      TIMESTAMP="$2"
      shift 2
      ;;
    -p | --print)
      print_config="print_config"
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

OUTPUT_DIR="${HOME}/imitation/output/train_experts/${TIMESTAMP}"

while IFS="," read -r env n_demos best_seed
do
  named_configs="search_space.named_configs=[\"${env}\"]"
  echo ${named_configs}
  python3 -m imitation.scripts.parallel ${print_config} with example_dagger seed=0 \
  ${named_configs} \
  base_config_updates.expert.policy_type="ppo" \
  base_config_updates.expert.loader_kwargs.path="${OUTPUT_DIR}/${env}_${best_seed}/policies/final/"
done < <(tail -n +2 ${CONFIG_CSV})

# shellcheck disable=SC2016
echo 'Generate results table using `python -m imitation.scripts.analyze`'
# python3 -m imitation.scripts.analyze analyze_imitation with source_dir_str=~/ray_results/dagger_tuning csv_output_path=logs_dagger_tuning.csv table_verbosity=-1