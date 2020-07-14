DEFAULT_INIT_RL_KWARGS = dict(
    # For recommended PPO hyperparams in each environment, see:
    # https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/ppo2.yml
    # (this was written for Stable Baselines 2, so it uses nminibatches and
    # n_steps rather than batch_size; values below should be equivalent)
    learning_rate=3e-4,
    batch_size=64,
    n_epochs=10,
    ent_coef=0.0,
)
