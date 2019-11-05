#!/usr/bin/env bash
# This script regenerates tests/data.
set -e


# Regenerate rollouts and policies
# experiments/train_experts.sh -r


# Regenerate tests/data/imit_benchmark (for analyze.analyze_imitation tests).
save_dir=tests/data/imit_benchmark
if [[ -d ${save_dir} ]]; then
  rm -r ${save_dir}
fi
mkdir -p ${save_dir}

tmp_dir="$(mktemp -d)"

alias imit_benchmark="experiments/imit_benchmark.sh -f --log_root ${tmp_dir} \
  --file_storage=${save_dir}"

shopt -s expand_aliases
imit_benchmark --airl --run_name FOO
imit_benchmark --gail --run_name BAR
imit_benchmark --gail --run_name BAR
