#!/usr/bin/env bash
set -e
source experiments/common.env

ENVS="cartpole"
DATA_DIR=${DATA_DIR:-"data"}
LOG_ROOT="output/dagger_benchmark/${TIMESTAMP}"
extra_configs=""
extra_options=""

SEEDS="0 1 2 3 4"

TEMP=$($GNU_GETOPT -o fT -l fast,paper,tmux,pdb,echo,run_name:,log_root:,file_storage: -- "$@")
if [[ $? != 0 ]]; then exit 1; fi
eval set -- "$TEMP"

while true; do
  case "$1" in
    # Fast mode (debug)
    -f | --fast)
      SEEDS="0"
      extra_configs+="fast "
      shift
      ;;
    --paper)  # Table benchmark settings
      ENVS="seals_cartpole seals_mountain_car half_cheetah"
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
      # Used by `tests/generate_test_data.sh` to save Sacred logs in tests/testdata.
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
  expert_policy_path=${DATA_DIR}/expert_models/{env_config_name}_0/policies/final/ \
  expert_policy_type='ppo' \
  seed={seed} \
  ${extra_configs} \
  ::: env_config_name ${ENVS} \
  ::: seed ${SEEDS}

# Directory path is really long. Enter the directory to shorten results output,
# which includes directory of each stdout file.
pushd ${LOG_ROOT}/parallel
find . -name stderr | sort | xargs tail -n 15 | grep -E '==|Result'
popd

echo 'Generate results table using `python -m imitation.scripts.analyze`'
