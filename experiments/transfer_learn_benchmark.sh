#!/usr/bin/env bash
# Train PPO2 experts using reward models from experiments/imit_benchmark.sh

TIMESTAMP=$(date --iso-8601=seconds)
REWARD_MODELS_DIR=reward_models
LOG_ROOT="output/train_experts/${TIMESTAMP}"
RESULTS_FILE="results.txt"
ALGORITHM="gail"
extra_configs=""


SEEDS="0 1 2"

ENVS+="acrobot cartpole mountain_car "
ENVS+="reacher half_cheetah hopper ant humanoid swimmer walker "
ENVS+="two_d_maze custom_ant disabled_ant "


TEMP=$(getopt -o f -l fast,gail,airl,run_name:,log_root: -- $@)
if [[ $? != 0 ]]; then exit 1; fi
eval set -- "$TEMP"

while true; do
  case "$1" in
    # Fast mode (debug)
    -f | --fast)
      ENVS="cartpole mountain_car"
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
      # Used by analysis scripts to filter runs later.
      extra_options+="--name $2 "
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

echo "Writing logs in ${LOG_ROOT}"
parallel -j 25% --header : --results ${LOG_ROOT}/parallel/ --progress \
  python -m imitation.scripts.expert_demos \
  ${extra_options} \
  with \
  {env_config_name} ${extra_configs} \
  seed={seed} \
  log_dir="${LOG_ROOT}/${ALGORITHM}/{env_config_name}_{seed}" \
  reward_type="DiscrimNet" \
  reward_path="${REWARD_MODELS_DIR}/${ALGORITHM}/{env_config_name}_0/checkpoints/final/discrim/" \
  ::: seed ${SEEDS} ::: env_config_name ${ENVS}


pushd $LOG_ROOT

# Display and save mean episode reward to ${RESULTS_FILE}.
find . -name stdout | xargs tail -n 15 | grep -E '(==|ep_reward_mean)' | tee ${RESULTS_FILE}

popd
