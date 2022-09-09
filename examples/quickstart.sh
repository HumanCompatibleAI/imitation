#!/usr/bin/env bash

# Train PPO agent on pendulum and collect expert demonstrations. Tensorboard logs saved in quickstart/rl/
python -m imitation.scripts.train_rl with pendulum common.fast train.fast rl.fast fast common.log_dir=quickstart/rl/

# Train GAIL from demonstrations. Tensorboard logs saved in output/ (default log directory).
python -m imitation.scripts.train_adversarial gail with pendulum common.fast demonstrations.fast train.fast rl.fast fast demonstrations.rollout_path=quickstart/rl/rollouts/final.pkl

# Train AIRL from demonstrations. Tensorboard logs saved in output/ (default log directory).
python -m imitation.scripts.train_adversarial airl with pendulum common.fast demonstrations.fast train.fast rl.fast fast demonstrations.rollout_path=quickstart/rl/rollouts/final.pkl
