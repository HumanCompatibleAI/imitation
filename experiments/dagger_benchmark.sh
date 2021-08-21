#!/usr/bin/env bash
set -e

ENVS="cartpole"
EXPERT_MODELS_DIR="data/expert_models"
TIMESTAMP=$(date --iso-8601=seconds)
LOG_ROOT="output/dagger_benchmark/${TIMESTAMP}"
extra_configs=""
extra_options=""

SEEDS="0 1 2 3 4"

TEMP=$(getopt -o fT -l fast,mvp,mvp_fast,tmux,pdb,echo,run_name:,log_root:,file_storage:,mvp_seals -- "$@")
if [[ $? != 0 ]]; then exit 1; fi
eval set -- "$TEMP"

while true; do
  case "$1" in
    # Fast mode (debug)
    -f | --fast)
      EXPERT_MODELS_DIR="tests/data/expert_models"
      SEEDS="0"
      extra_configs+="fast "
      shift
      ;;
    --mvp_seals)  # Table benchmark settings
      ENVS="seals_cartpole seals_mountain_car half_cheetah"
      shift
      ;;
    --mvp_fast)  # Debug or quickly validate the benchmark settings
      ENVS="seals_cartpole seals_mountain_car half_cheetah"
      SEEDS="0"
      extra_configs+="fast "
      shift
      ;;
    -T | --tmux)
      extra_parallel_options+="--tmux "
      shift
      ;;
    --run_name)
      extra_options+="--name $2 "
      shift 2
      ;;
    --log_root)
      LOG_ROOT="$2"
      shift 2
      ;;
    --file_storage)
      # Used by `tests/generate_test_data.sh` to save Sacred logs in tests/data.
      extra_options+="--file_storage $2 "
      shift 2
      ;;
    --pdb)
      echo 'NOTE: Interact with PDB session via tmux. If an error occurs, `parallel` '
      echo 'will hang and wait for user input in tmux session.'
      extra_parallel_options+="--tmux "  # Needed for terminal output.
      extra_options+="--pdb "
      shift
      ;;
    --echo)
      extra_parallel_options+="echo "
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

parallel -j 25% --header : --results ${LOG_ROOT}/parallel/ --colsep , --progress \
  ${extra_parallel_options} \
  python -m imitation.scripts.train_dagger \
  --capture=sys \
  ${extra_options} \
  with \
  {env_config_name} \
  log_dir="${LOG_ROOT}/{env_config_name}_{seed}" \
  expert_data_src_format=None \
  expert_policy_path=${EXPERT_MODELS_DIR}/{env_config_name}_0/policies/final/ \
  expert_policy_type='ppo' \
  seed={seed} \
  ${extra_configs} \
  ::: env_config_name ${ENVS} \
  ::: seed ${SEEDS}

# Directory path is really long. Enter the directory to shorten results output,
# which includes directory of each stdout file.
pushd ${LOG_ROOT}/parallel
find . -name stdout | sort | xargs tail -n 15 | grep -E '==|\[result\]'
popd

echo 'Generate results table using `python -m imitation.scripts.analyze`'
