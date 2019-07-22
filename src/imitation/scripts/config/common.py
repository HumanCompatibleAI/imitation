DEFAULT_BLANK_POLICY_KWARGS = dict(
    learning_rate=3e-4,
    nminibatches=32,
    noptepochs=10,
    # WARNING: this is actually 8*256=2048 due to 8 vector environments
    n_steps=256,
)
