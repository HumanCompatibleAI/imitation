#!/usr/bin/env bash

# Train PPO agent on pendulum and collect expert demonstrations. Tensorboard logs saved in quickstart/rl/
python -m imitation.scripts.train_rl with pendulum environment.fast policy_evaluation.fast rl.fast fast logging.log_dir=quickstart/rl/

# Train GAIL from demonstrations. Tensorboard logs saved in output/ (default log directory).
python -m imitation.scripts.train_adversarial gail with pendulum environment.fast demonstrations.fast policy_evaluation.fast rl.fast fast demonstrations.path=quickstart/rl/rollouts/final.npz

# Train AIRL from demonstrations. Tensorboard logs saved in output/ (default log directory).
python -m imitation.scripts.train_adversarial airl with pendulum environment.fast demonstrations.fast policy_evaluation.fast rl.fast fast demonstrations.path=quickstart/rl/rollouts/final.npz
