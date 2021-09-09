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
    env_make_kwargs = {}  # The kwargs passed to `spec.make`.

    iterations = 10
    agent_steps = 1e4
    sample_steps = 1e4
    fragment_length = 10
    num_pairs = 100
    n_episodes_eval = 50  # Num of episodes for final mean ground truth return
    reward_kwargs = {}
    agent_kwargs = {}
    allow_variable_horizon = False

    # Number of environments in VecEnv
    num_vec = 8

    # Use SubprocVecEnv rather than DummyVecEnv (generally faster if num_vec>1)
    parallel = True
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
