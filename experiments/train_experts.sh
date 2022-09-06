#!/usr/bin/env bash
set -e

# This script trains experts for experiments/imit_benchmark.sh.
# When training is finished, it reports the mean episode reward of each
# expert.

source experiments/common.sh

SEEDS=(0 1 2 3 4)
ENVS=(seals_ant seals_half_cheetah)

OUTPUT_DIR="output/train_experts/${TIMESTAMP}"
RESULTS_FILE="results.txt"
extra_configs=()
print_config=false

if ! TEMP=$($GNU_GETOPT -o frw -l fast,regenerate,wandb,print -- "$@"); then
  exit 1
fi
eval set -- "$TEMP"

while true; do
  case "$1" in
    -f | --fast)
      # Fast mode (debug)
      ENVS=(cartpole pendulum)
      SEEDS=(0)
      extra_configs=("${extra_configs[@]}" common.fast rl.fast train.fast fast)
      shift
      ;;
    -w | --wandb)
      # activate wandb logging by adding 'wandb' format string to common.log_format_strs
      extra_configs=("${extra_configs[@]}" "common.wandb_logging" "common.wandb.wandb_kwargs.project='algorithm-benchmark' ")
      shift
      ;;
    -r | --regenerate)
      # Regenerate test data (policies and rollouts).
      #
      # Combine with fast mode flag to generate low-computation versions of
      # test data.
      # Use `git clean -df tests/testdata` to remove extra log files.
      ENVS=(cartpole pendulum)
      SEEDS=(0)
      OUTPUT_DIR="tests/testdata/expert_models"
      extra_configs=("${extra_configs[@]}" "rollout_save_n_episodes=50")

      if [[ -d ${OUTPUT_DIR} ]]; then
        rm -r ${OUTPUT_DIR}
      fi

      shift
      ;;
    -p | --print)
      print_config=true
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

if [ "$print_config" = true ]; then
  parallel -j 25% --header : --progress --results ${OUTPUT_DIR}/parallel/ \
    python3 -m imitation.scripts.train_rl print_config \
    --capture=sys \
    with \
    '{env}' "${extra_configs[@]}" \
    seed='{seed}' \
    common.log_dir="${OUTPUT_DIR}/{env}_{seed}" \
    ::: env "${ENVS[@]}" \
    ::: seed "${SEEDS[@]}"
    exit 0
fi



echo "Writing logs in ${OUTPUT_DIR}"
# Train experts.
parallel -j 25% --header : --progress --results ${OUTPUT_DIR}/parallel/ \
  python3 -m imitation.scripts.train_rl \
  --capture=sys \
  with \
  '{env}' "${extra_configs[@]}" \
  seed='{seed}' \
  common.log_dir="${OUTPUT_DIR}/{env}_{seed}" \
  ::: env "${ENVS[@]}" \
  ::: seed "${SEEDS[@]}"

pushd "$OUTPUT_DIR"

# Display and save mean episode reward to ${RESULTS_FILE}.
find . -name stdout -print0 | sort -z | xargs -0 tail -n 15 | grep -E '(==|return_mean)' | tee ${RESULTS_FILE}

popd

echo "[Optional] Upload new experts to S3 (replacing old ones) using the commands:"
echo "aws s3 rm --recursive s3://shwang-chai/public/data/expert_models"
echo "aws s3 sync --exclude '*/rollouts/*' --exclude '*/policies/*' --include '*/policies/final/*' '${OUTPUT_DIR}' s3://shwang-chai/public/data/expert_models"
