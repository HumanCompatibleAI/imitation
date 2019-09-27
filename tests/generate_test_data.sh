#!/usr/bin/env bash
# This script regenerates tests/data.


# Regenerate rollouts and policies
experiments/train_experts.sh -r


# Regenerate tests/data/sacred (for analysis tests).
tmp_dir="$(mktemp -d)"
experiments/download_experts.sh
experiments/imit_benchmark.sh -f --airl --run_name FOO --log_dir "${tmp_dir}/1"
experiments/imit_benchmark.sh -f --gail --run_name BAR --log_dir "${tmp_dir}/2"
experiments/imit_benchmark.sh -f --gail --run_name BAR --log_dir "${tmp_dir}/3"

save_dir=tests/data/sacred/imit_benchmark
if [[ -d ${save_dir} ]]; then
  rm -r ${save_dir}
fi

rm -r tests/data/sacred/imit_benchmark
for i in 1 2 3; do
  sacred_path=$(readlink "${tmp_dir}/${i}")
  cp -vR "${tmp_dir}/${i}/sacred/" "${save_dir}/${i}/"
