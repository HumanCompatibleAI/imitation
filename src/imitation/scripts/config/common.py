DEFAULT_INIT_RL_KWARGS = dict(
    # For recommended PPO hyperparams in each environment, see:
    # https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/ppo2.yml
    learning_rate=3e-4,
    nminibatches=32,
    noptepochs=10,
    ent_coef=0.0,
)
