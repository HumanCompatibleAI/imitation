DEFAULT_INIT_RL_KWARGS = dict(
    # For recommended PPO hyperparams in each environment, see:
    # https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/ppo2.yml
    learning_rate=3e-4,
    nminibatches=32,
    noptepochs=10,
    # WARNING: this is actually 8*256=2048 due to 8 vector environments
    n_steps=256,
    ent_coef=0.0,
)
