import sacred

from imitation.scripts.config.common import DEFAULT_BLANK_POLICY_KWARGS
from imitation.util import FeedForward64Policy

train_ex = sacred.Experiment("train", interactive=True)

_init_trainer_kwargs = {}


@train_ex.config
def train_defaults():
    n_epochs = 50
    n_disc_steps_per_epoch = 50
    n_gen_steps_per_epoch = 2048

    init_trainer_kwargs = _init_trainer_kwargs
    init_trainer_kwargs.update(dict(
        use_random_expert=False,
        num_vec=8,  # NOTE: changing this also changes the effective nsteps!
        reward_kwargs=dict(
            theta_units=[32, 32],
            phi_units=[32, 32],
        ),

        trainer_kwargs=dict(
            n_disc_samples_per_buffer=1000,
            # Setting buffer capacity and disc samples to 1000 effectively
            # disables the replay buffer. This seems to improve convergence
            # speed, but may come at a cost of stability.
            gen_replay_buffer_capacity=1000,
            n_expert_samples=1000,
        ),

        # Some environments (e.g. CartPole) have float max as limits, which
        # breaks the scaling.
        discrim_scale=False,

        make_blank_policy_kwargs=DEFAULT_BLANK_POLICY_KWARGS
    ))


@train_ex.named_config
def gail():
    init_trainer_kwargs = _init_trainer_kwargs
    init_trainer_kwargs.update(dict(
        use_gail=True,
    ))


@train_ex.named_config
def airl():
    init_trainer_kwargs = _init_trainer_kwargs
    init_trainer_kwargs.update(dict(
        use_gail=False,
    ))


@train_ex.named_config
def ant():
    env_name = "Ant-v2"
    n_epochs = 2000


@train_ex.named_config
def cartpole():
    env_name = "CartPole-v1"


@train_ex.named_config
def halfcheetah():
    env_name = "HalfCheetah-v2"
    n_epochs = 1000

    init_trainer_kwargs = _init_trainer_kwargs
    init_trainer_kwargs.update(dict(
        discrim_kwargs=dict(entropy_weight=0.1),
    ))


@train_ex.named_config
def pendulum():
    env_name = "Pendulum-v0"


@train_ex.named_config
def swimmer():
    env_name = "Swimmer-v2"
    n_epochs = 1000
    init_trainer_kwargs = _init_trainer_kwargs
    init_trainer_kwargs["make_blank_policy_kwargs"].update(dict(
        policy_network_class=FeedForward64Policy
    ))


@train_ex.named_config
def debug():
    n_epochs = 1
    interactive = False
    n_disc_steps_per_epoch = 1
    n_gen_steps_per_epoch = 1
    n_episodes_per_reward_data = 1
