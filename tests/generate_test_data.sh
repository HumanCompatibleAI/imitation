#!/usr/bin/env bash
# This script regenerates tests/testdata.
set -e

# Regenerate tests/testdata/expert_models (for various tests).
experiments/train_experts.sh -r

mkdir -p tests/testdata/expert_models/cartpole_0/policies/final_without_vecnorm
ln -sf ../final/model.zip tests/testdata/expert_models/cartpole_0/policies/final_without_vecnorm/model.zip
