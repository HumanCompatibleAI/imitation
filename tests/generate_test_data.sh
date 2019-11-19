#!/usr/bin/env bash
# This script regenerates tests/data.
set -e

# Regenerate tests/data/expert_models (for various tests).
experiments/train_experts.sh -r
