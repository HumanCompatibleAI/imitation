#!/usr/bin/env bash
# Train PPO experts using reward models from experiments/imit_benchmark.sh

set -e

source experiments/common.sh

SEEDS=(0 1 2)
CONFIG_CSV="experiments/imit_benchmark_config.csv"
REWARD_MODELS_DIR="data/reward_models"
# To prevent race conditions, we use a different log root for each process id.
LOG_ROOT="output/train_experts/${TIMESTAMP}-${BASHPID}"
RESULTS_FILE="results.txt"
ALGORITHM="gail"
NEED_TEST_FILES="false"
extra_configs=()


if ! TEMP=$($GNU_GETOPT -o fw -l fast,gail,airl,run_name:,log_root:,wandb -- "$@"); then
  exit 1
fi
eval set -- "$TEMP"

while true; do
  case "$1" in
    # Fast mode (debug)
    -f | --fast)
      CONFIG_CSV="tests/testdata/imit_benchmark_config.csv"
      REWARD_MODELS_DIR="tests/testdata/reward_models"
      NEED_TEST_FILES="true"
      SEEDS=(0)
      extra_configs=("${extra_configs[@]}" environment.fast rl.fast policy_evaluation.fast fast)
      shift
      ;;
    -w | --wandb)
      # activate wandb logging by adding 'wandb' format string to logging.log_format_strs
      extra_configs=("${extra_configs[@]}" "logging.wandb_logging")
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
      # Used by analysis scripts to filter runs later.
      extra_options=("${extra_options[@]}" --name "$2")
      shift 2
      ;;
    --log_root)
      LOG_ROOT="$2"
      shift 2
      ;;
    --)
      shift
      break
      ;;
    *)
      echo "Unrecognized argument $1"
      exit 1
      ;;
  esac
done


if [[ $NEED_TEST_FILES == "true" ]]; then
  # Generate quick reward models for test.
  # To prevent race conditions, we use a different save_dir for each process id.
  save_dir=tests/testdata/reward_models/${ALGORITHM}/${TIMESTAMP}-${BASHPID}

  # Wipe directories for writing later.
  if [[ -d ${save_dir} ]]; then
    rm -r "${save_dir}"
  fi
  mkdir -p "${save_dir}"

  experiments/imit_benchmark.sh -f --${ALGORITHM} --log_root "${save_dir}"
fi


echo "Writing logs in ${LOG_ROOT}"
parallel -j 25% --header : --results "${LOG_ROOT}/parallel/" --colsep , --progress \
  python -m imitation.scripts.train_rl \
  --capture=sys \
  "${extra_options[@]}" \
  with \
  '{env_config_name}' seed='{seed}' \
  logging.log_dir="${LOG_ROOT}/${ALGORITHM}/{env_config_name}_{seed}/n_expert_demos_{n_expert_demos}" \
  reward_type="RewardNet_unshaped" \
  reward_path="${REWARD_MODELS_DIR}/${ALGORITHM}/${TIMESTAMP}-${BASHPID}/{env_config_name}_0/n_expert_demos_{n_expert_demos}/checkpoints/final/reward_test.pt" \
  "${extra_configs[@]}" \
  :::: ${CONFIG_CSV} \
  ::: seed "${SEEDS[@]}"

pushd "$LOG_ROOT"

# Display and save mean episode reward to ${RESULTS_FILE}.
find . -name stdout -print0 | sort -z | xargs -0 tail -n 15 | grep -E '(==|Result)' | tee ${RESULTS_FILE}

popd
