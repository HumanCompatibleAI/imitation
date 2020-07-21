DEFAULT_INIT_RL_KWARGS = dict(
    # For recommended PPO hyperparams in each environment, see:
    # https://github.com/DLR-RM/rl-baselines3-zoo/blob/master/hyperparams/ppo.yml
    learning_rate=3e-4,
    batch_size=64,
    n_epochs=10,
    ent_coef=0.0,
)
