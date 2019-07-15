"""Configuration for imitation.scripts.train."""

import os

import sacred

from imitation import util
from imitation.scripts.config.common import DEFAULT_BLANK_POLICY_KWARGS

train_ex = sacred.Experiment("train", interactive=True)


@train_ex.config
def train_defaults():
    env_name = "CartPole-v1"  # environment to train on
    n_epochs = 50
    n_disc_steps_per_epoch = 50
    n_gen_steps_per_epoch = 2048

    init_trainer_kwargs = dict(
        use_random_expert=False,
        num_vec=8,  # NOTE: changing this also changes the effective n_steps!
        parallel=True,  # Use SubprocVecEnv (generally faster if num_vec>1)
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

        make_blank_policy_kwargs=DEFAULT_BLANK_POLICY_KWARGS,
    )

    checkpoint_interval = 5  # number of epochs at which to checkpoint


@train_ex.config
def logging(env_name):
    log_dir = os.path.join("output", env_name.replace('/', '_'),
                           util.make_timestamp())


@train_ex.named_config
def gail():
    init_trainer_kwargs = dict(
        use_gail=True,
    )


@train_ex.named_config
def airl():
    init_trainer_kwargs = dict(
        use_gail=False,
    )


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

    init_trainer_kwargs = dict(
        discrim_kwargs=dict(entropy_weight=0.1),
    )


@train_ex.named_config
def pendulum():
    env_name = "Pendulum-v0"


@train_ex.named_config
def swimmer():
    env_name = "Swimmer-v2"
    n_epochs = 1000
    init_trainer_kwargs = dict(
        make_blank_policy_kwargs=dict(
            policy_network_class=util.FeedForward64Policy,
        ),
    )


@train_ex.named_config
def debug():
    n_epochs = 1
    interactive = False
    n_disc_steps_per_epoch = 1
    n_gen_steps_per_epoch = 1
    n_episodes_per_reward_data = 1
    init_trainer_kwargs = dict(
        parallel=False,  # easier to debug with everything in one process
    )
