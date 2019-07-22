DEFAULT_BLANK_POLICY_KWARGS = dict(
    # For recommended PPO hyperparams in each environment, see:
    # https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/ppo2.yml
    learning_rate=3e-4,
    nminibatches=32,
    noptepochs=10,
    # WARNING: this is actually 8*2048=16384 due to 8 vector environments
    # (n_steps=2048 across 8 vector environments is also what the Zoo uses)
    n_steps=2048,
)
