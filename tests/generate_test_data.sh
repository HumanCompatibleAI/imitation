#!/usr/bin/env bash
# This script regenerates tests/testdata.
set -e

# Regenerate tests/testdata/expert_models (for various tests).
experiments/train_experts.sh -r
