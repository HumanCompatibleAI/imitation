#!/usr/bin/env bash
set -e

# This script trains BC experts.
#
# Use the --paper flag to produce paper benchmark results.
#
# When training is finished, it reports the mean episode reward of each
# expert.
source experiments/common.sh

ENVS=(seals_cartpole)
SEEDS=(0 1 2)
# To prevent race conditions, we use a different output dir for each process id.
OUTPUT_DIR="output/bc_benchmark/${TIMESTAMP}-${BASHPID}"
extra_configs=()
extra_options=()
extra_parallel_options=()

if ! TEMP=$($GNU_GETOPT -o fTw -l fast,paper,tmux,run_name:,wandb -- "$@"); then
  exit 1
fi
eval set -- "$TEMP"

while true; do
  case "$1" in
    -f | --fast)
      # Fast mode (debug)
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
    --run_name)
      extra_options=("${extra_options[@]}" --name "$2")
      shift 2
      ;;
    -T | --tmux)
      extra_parallel_options=("${extra_parallel_options[@]}" --tmux)
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

echo "Writing logs in ${OUTPUT_DIR}"

parallel -j 25% --header : --results "${OUTPUT_DIR}/parallel/" --colsep , --progress \
  "${extra_parallel_options[@]}" \
  python -m imitation.scripts.train_imitation \
  --capture=sys \
  "${extra_options[@]}" \
  bc \
  with \
  '{env_config_name}' \
  "${extra_configs[@]}" \
  'seed={seed}' \
  logging.log_root="${OUTPUT_DIR}" \
  ::: env_config_name "${ENVS[@]}" \
  ::: seed "${SEEDS[@]}"
