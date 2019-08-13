from imitation.policies.base import MnistCnnPolicy

DEFAULT_BLANK_POLICY_KWARGS = dict(
    # For recommended PPO hyperparams in each environment, see:
    # https://github.com/araffin/rl-baselines-zoo/blob/master/hyperparams/ppo2.yml
    learning_rate=3e-4,
    nminibatches=32,
    noptepochs=10,
    # WARNING: this is actually 8*256=2048 due to 8 vector environments
    n_steps=256,
    ent_coef=0.0,
)

ATARI_CNN_BLANK_POLICY_KWARGS = dict(
    n_steps=128,
    noptepochs=4,
    nminibatches=4,
    learning_rate=2.5e-4,
    cliprange=0.1,
    vf_coef=0.5,
    ent_coef=0.01,
    cliprange_vf=-1,
    policy_class=MnistCnnPolicy,
)
