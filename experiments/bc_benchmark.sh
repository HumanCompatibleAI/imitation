#!/usr/bin/env bash
set -e

# This script trains BC experts.
#
# Use the --paper flag to produce paper benchmark results.
#
# When training is finished, it reports the mean episode reward of each
# expert.
source experiments/common.sh

ENVS=(cartpole)
SEEDS=(0 1 2)
DATA_DIR=${DATA_DIR:-"data/"}
OUTPUT_DIR="output/bc_benchmark/${TIMESTAMP}"
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
      extra_configs=("${extra_configs[@]}" common.fast demonstrations.fast train.fast fast)
      DATA_DIR="tests/testdata"
      shift
      ;;
    --paper)  # Table benchmark settings
      ENVS=(seals_cartpole seals_mountain_car half_cheetah)
      shift
      ;;
    -w | --wandb)
      # activate wandb logging by adding 'wandb' format string to common.log_format_strs
      extra_configs=("${extra_configs[@]}" "common.wandb_logging")
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
  common.log_root="${OUTPUT_DIR}" \
  demonstrations.rollout_path="${DATA_DIR}/expert_models/{env_config_name}_0/rollouts/final.pkl" \
  ::: env_config_name "${ENVS[@]}" \
  ::: seed "${SEEDS[@]}"
