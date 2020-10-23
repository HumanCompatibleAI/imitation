# Train PPO agent on cartpole and collect expert demonstrations. Tensorboard logs saved in `quickstart/rl/`
python -m imitation.scripts.expert_demos with fast cartpole log_dir=quickstart/rl/

# Train GAIL from demonstrations. Tensorboard logs saved in output/ (default log directory).
python -m imitation.scripts.train_adversarial with fast gail cartpole rollout_path=quickstart/rl/rollouts/final.pkl

# Train AIRL from demonstrations. Tensorboard logs saved in output/ (default log directory).
python -m imitation.scripts.train_adversarial with fast airl cartpole rollout_path=quickstart/rl/rollouts/final.pkl
