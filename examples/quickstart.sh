# Train PPO agent on pendulum and collect expert demonstrations. Tensorboard logs saved in `quickstart/rl/`
python -m imitation.scripts.expert_demos with pendulum fast log_dir=quickstart/rl/

# Train GAIL from demonstrations. Tensorboard logs saved in output/ (default log directory).
python -m imitation.scripts.train_adversarial with gail pendulum fast rollout_path=quickstart/rl/rollouts/final.pkl

# Train AIRL from demonstrations. Tensorboard logs saved in output/ (default log directory).
python -m imitation.scripts.train_adversarial with airl pendulum fast rollout_path=quickstart/rl/rollouts/final.pkl
