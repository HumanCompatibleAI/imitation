#!/usr/bin/env bash

# PREREQUISITE: Run experiments/download_models.sh to get rollouts.
#
# You can find expected results for GAIL and AIRL here:
#   https://github.com/HumanCompatibleAI/imitation/pull/317
#
# This end-to-end script runs all the scripts necessary to generate the paper
# results, and then generates TeX and CSV tables for the experiment output
# in a variety of table verbosities.
#
# To test this script out, run with the --fast flag.
#
# You can reduce stdout verbosity of the imitation runs and view imitation
# progress more easily by using the --tmux or -T flag. However, this flag
# may hide errors from your terminal, so it is not recommended during debugging.
#
# All imitation runs are given the same timestamped `--run_name` so that
# they can be gathered by the table-generating analysis script.

set -e  # Exit on error
source experiments/common.sh

RUN_NAME="paper-${TIMESTAMP}"
echo "Training with run_name=${RUN_NAME}"

script_dir=experiments
fast_flag=()
paper_flag=("--paper")
tmux_flag=()

if ! TEMP=$($GNU_GETOPT -o fT -l fast,tmux -- "$@"); then
  exit 1
fi
eval set -- "$TEMP"

while true; do
  case "$1" in
    -f | --fast)
      # Use this flag to quickly test a shortened benchmark and table
      fast_flag=("--fast")
      paper_flag=()
      # To prevent race conditions, we use a different run name for each process id.
      RUN_NAME="test-${TIMESTAMP}-$BASHPID"
      shift
      ;;
    -T | --tmux)
      tmux_flag=("--tmux")
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


set -ex  # Start echoing commands

echo "BC BENCHMARK"
${script_dir}/bc_benchmark.sh "${fast_flag[@]}" "${paper_flag[@]}" "${tmux_flag[@]}" --run_name "$RUN_NAME"

IMIT_PLAIN="${script_dir}/imit_benchmark.sh ${fast_flag[*]} ${tmux_flag[*]} --run_name $RUN_NAME"

echo "AIRL seals BENCHMARK"
$IMIT_PLAIN --mvp_seals --airl

echo "GAIL seals BENCHMARK"
$IMIT_PLAIN --mvp_seals --gail

if [ ${#fast_flag[@]} -eq 0 ]; then
  # Fast flag not specified.
  echo "AIRL/GAIL HalfCheetah BENCHMARK"
  $IMIT_PLAIN --cheetah
fi

echo "DAGGER BENCHMARK"
${script_dir}/dagger_benchmark.sh "${fast_flag[@]}" "${paper_flag[@]}" "${tmux_flag[@]}" --run_name "$RUN_NAME"

result_dir=output/fast_table_result
mkdir -p $result_dir
for v in 0 1 2; do
  base_out_path=$result_dir/fast_table_result_verbosity$v

  python -m imitation.scripts.analyze analyze_imitation with \
    source_dir_str="output/sacred" table_verbosity=$v  \
    csv_output_path=$base_out_path.csv \
    tex_output_path=$base_out_path.tex \
    run_name="$RUN_NAME"
done
