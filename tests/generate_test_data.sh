#!/usr/bin/env bash
# This script regenerates tests/data.
set -e


# Regenerate rollouts and policies
experiments/train_experts.sh -r


# Regenerate tests/data/imit_benchmark (for analysis tests).
tmp_dir="$(mktemp -d)"
experiments/imit_benchmark.sh -f --airl --run_name FOO --log_root "${tmp_dir}/1"
experiments/imit_benchmark.sh -f --gail --run_name BAR --log_root "${tmp_dir}/2"
experiments/imit_benchmark.sh -f --gail --run_name BAR --log_root "${tmp_dir}/3"

save_dir=tests/data/imit_benchmark
if [[ -d ${save_dir} ]]; then
  rm -r ${save_dir}
fi
mkdir -p ${save_dir}
ls ${tmp_dir}
cp -RLv ${tmp_dir}/*/ "${save_dir}"
