#!/usr/bin/env bash
# This script regenerates tests/testdata.
set -e

# Regenerate tests/testdata/expert_models (for various tests).
experiments/train_experts.sh -r

# Gather preferences (for preference comparison test).
# We only gather a minimal dataset for smoke tests.
# We load existing trajectories and set the reward trainer epochs to zero
# so that only preferences need to be gathered.
python -m imitation.scripts.train_preference_comparisons with \
    cartpole \
    trajectory_path=tests/testdata/expert_models/cartpole_0/rollouts/final.pkl \
    total_comparisons=10 \
    comparisons_per_iteration=10 \
    reward_trainer_kwargs.epochs=0 \
    fragment_length=2 \
    save_preferences=True \
    log_dir=tests/testdata/preferences/cartpole