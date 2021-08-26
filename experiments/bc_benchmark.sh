#!/usr/bin/env bash
set -e

# This script trains BC experts.
#
# Use the --mvp_seals flag to produce paper benchmark results.
#
# TODO(shwang): Just rename mvp_seals flag in this file and others to --paper
#
# When training is finished, it reports the mean episode reward of each
# expert.
source experiments/common.env

ENVS+="cartpole"
SEEDS="0 1 2"

DATA_DIR=${DATA_DIR:-"data/"}
OUTPUT_DIR="output/bc_benchmark/${TIMESTAMP}"
RESULTS_FILE="results.txt"
extra_configs=""

TEMP=$($GNU_GETOPT -o fT -l fast,tmux,run_name:,mvp_seals,mvp_fast -- "$@")
if [[ $? != 0 ]]; then exit 1; fi
eval set -- "$TEMP"

while true; do
  case "$1" in
    -f | --fast)
      # Fast mode (debug)
      SEEDS="0"
      extra_configs+="fast "
      shift
      ;;
    --mvp_seals)  # Table benchmark settings
      ENVS="seals_cartpole seals_mountain_car half_cheetah "
      shift
      ;;
    --run_name)
      extra_options+="--name $2 "
      shift 2
      ;;
    -T | --tmux)
      extra_parallel_options+="--tmux "
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

parallel -j 25% --header : --results ${OUTPUT_DIR}/parallel/ --colsep , --progress \
  ${extra_parallel_options} \
  python -m imitation.scripts.train_bc \
  --capture=sys \
  ${extra_options} \
  with \
  {env_cfg_name} \
  ${extra_configs} \
  expert_data_src=${DATA_DIR}/expert_models/{env_cfg_name}_0/rollouts/final.pkl \
  expert_data_src_format="path" \
  seed={seed} \
  log_root=${OUTPUT_DIR} \
  ::: env_cfg_name ${ENVS} \
  ::: seed ${SEEDS}

pushd $OUTPUT_DIR

popd
