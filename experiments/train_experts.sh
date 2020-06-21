#!/usr/bin/env bash
set -e

# This script trains experts for experiments/imit_benchmark.sh.
# When training is finished, it reports the mean episode reward of each
# expert.

ENVS+="acrobot cartpole mountain_car "
ENVS+="reacher half_cheetah hopper ant humanoid swimmer walker "
ENVS+="two_d_maze custom_ant disabled_ant "

SEEDS="0 1 2"

TIMESTAMP=$(date --iso-8601=seconds)
OUTPUT_DIR="output/train_experts/${TIMESTAMP}"
RESULTS_FILE="results.txt"
extra_configs=""

TEMP=$(getopt -o fr -l fast,regenerate -- $@)
if [[ $? != 0 ]]; then exit 1; fi
eval set -- "$TEMP"

while true; do
  case "$1" in
    -f | --fast)
      # Fast mode (debug)
      ENVS="cartpole pendulum"
      SEEDS="0"
      extra_configs+="fast "
      shift
      ;;
    -r | --regenerate)
      # Regenerate test data (policies and rollouts).
      #
      # Combine with fast mode flag to generate low-computation versions of
      # test data.
      # Use `git clean -df tests/data` to remove extra log files.
      ENVS="cartpole pendulum"
      SEEDS="0"
      OUTPUT_DIR="tests/data/expert_models"
      extra_configs+="rollout_save_n_episodes=50 "

      if [[ -d ${OUTPUT_DIR} ]]; then
        rm -r ${OUTPUT_DIR}
      fi

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
# Train experts.
parallel -j 25% --header : --progress --results ${OUTPUT_DIR}/parallel/ \
  python -m imitation.scripts.expert_demos \
  --capture=sys \
  with \
  {env} ${extra_configs} \
  seed={seed} \
  log_dir="${OUTPUT_DIR}/{env}_{seed}" \
  ::: env ${ENVS} ::: seed ${SEEDS}

pushd $OUTPUT_DIR

# Display and save mean episode reward to ${RESULTS_FILE}.
find . -name stdout | xargs tail -n 15 | grep -E '(==|ep_reward_mean)' | tee ${RESULTS_FILE}

popd

echo "[Optional] Upload new experts to S3 (replacing old ones) using the commands:"
echo "aws s3 rm --recursive s3://shwang-chai/public/data/expert_models"
echo "aws s3 sync --exclude '*/rollouts/*' --exclude '*/policies/*' --include '*/policies/final/*' '${OUTPUT_DIR}' s3://shwang-chai/public/data/expert_models"
