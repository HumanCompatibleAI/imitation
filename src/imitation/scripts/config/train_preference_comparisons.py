"""Configuration for imitation.scripts.train_preference_comparisons."""

import os

import sacred

from imitation.util import util

train_preference_comparisons_ex = sacred.Experiment(
    "train_preference_comparisons", interactive=True
)


@train_preference_comparisons_ex.config
def train_defaults():
    env_name = "seals/CartPole-v0"  # environment to train on
    iterations = 10
    agent_steps = 1e4
    sample_steps = 1e4
    fragment_length = 100
    num_pairs = 20
    n_episodes_eval = 50  # Num of episodes for final mean ground truth return
    reward_net_kwargs = {}
    reward_trainer_kwargs = {
        "epochs": 3,
    }
    agent_kwargs = {}
    gatherer_kwargs = {}
    trajectory_path = None
    allow_variable_horizon = False

    # TODO(ejnnr): Set to 1 mostly do speed up experimentation, should be increased
    # Number of environments in VecEnv
    num_vec = 1

    # TODO(ejnnr): should probably be set to True again once num_vec is increased
    # Use SubprocVecEnv rather than DummyVecEnv (generally faster if num_vec>1)
    parallel = False
    max_episode_steps = None  # Set to positive int to limit episode horizons

    log_root = os.path.join(
        "output", "train_preference_comparisons"
    )  # output directory


@train_preference_comparisons_ex.config
def paths(env_name, log_root):
    log_dir = os.path.join(
        log_root, env_name.replace("/", "_"), util.make_unique_timestamp()
    )


@train_preference_comparisons_ex.named_config
def cartpole():
    env_name = "CartPole-v1"
    allow_variable_horizon = True


@train_preference_comparisons_ex.named_config
def seals_cartpole():
    env_name = "seals/CartPole-v0"


@train_preference_comparisons_ex.named_config
def mountain_car():
    env_name = "MountainCar-v0"
    allow_variable_horizon = True


# Debug configs


@train_preference_comparisons_ex.named_config
def fast():
    """Minimize the amount of computation.

    Useful for test cases.
    """
    iterations = 1
    agent_steps = 10
    sample_steps = 10
    agent_steps = 2
    parallel = False
    num_vec = 1
    fragment_length = 2
    num_pairs = 2
    n_episodes_eval = 1
    agent_kwargs = {"batch_size": 2, "n_steps": 10, "n_epochs": 1}
