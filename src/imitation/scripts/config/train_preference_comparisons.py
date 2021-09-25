"""Configuration for imitation.scripts.train_preference_comparisons."""

import os

import sacred

from imitation.envs import maze
from imitation.util import util

train_preference_comparisons_ex = sacred.Experiment(
    "train_preference_comparisons",
    interactive=True,
)

maze.use_config(train_preference_comparisons_ex)


@train_preference_comparisons_ex.config
def train_defaults():
    env_name = "seals/CartPole-v0"  # environment to train on
    env_make_kwargs = {}  # The kwargs passed to `spec.make`.

    fragment_length = 20  # timesteps per fragment used for comparisons
    total_timesteps = int(1e6)  # total number of environment timesteps
    total_comparisons = 5000  # total number of comparisons to elicit
    # comparisons to gather before switching back to agent training
    comparisons_per_iteration = 300
    # factor by which to oversample transitions before creating fragments
    transition_oversampling = 1
    initial_comparison_fraction = 0.1
    random_frac = 0.0

    n_episodes_eval = 50  # Num of episodes for final mean ground truth return
    reward_net_kwargs = {"use_action": False}
    reward_trainer_kwargs = {
        "epochs": 3,
    }
    save_preferences = False  # save preference dataset at the end?
    agent_path = None  # path to a (partially) trained agent to load at the beginning
    agent_kwargs = {}
    gatherer_kwargs = {}
    # path to a pickled sequence of trajectories used instead of training an agent
    trajectory_path = None
    value_net_path = None
    allow_variable_horizon = False

    num_vec = 8  # number of parallel environments

    normalize = True  # Use VecNormalize
    normalize_kwargs = {"norm_reward": False}  # kwargs for `VecNormalize`

    # Use SubprocVecEnv rather than DummyVecEnv (generally faster if num_vec>1)
    parallel = True
    max_episode_steps = None  # Set to positive int to limit episode horizons

    log_root = os.path.join(
        "output",
        "train_preference_comparisons",
    )  # output directory


@train_preference_comparisons_ex.config
def paths(env_name, log_root):
    log_dir = os.path.join(
        log_root,
        env_name.replace("/", "_"),
        util.make_unique_timestamp(),
    )

@train_preference_comparisons_ex.named_config
def good(fragment_length, total_comparisons, comparisons_per_iteration):
    fragment_length = fragment_length // 2
    total_comparisons *= 2
    comparisons_per_iteration *= 2
    gatherer_kwargs = {"sample": False}

@train_preference_comparisons_ex.named_config
def medium():
    gatherer_kwargs = {"sample": False}

@train_preference_comparisons_ex.named_config
def bad(fragment_length, total_comparisons, comparisons_per_iteration):
    fragment_length *= 4
    total_comparisons = total_comparisons // 2
    comparisons_per_iteration = comparisons_per_iteration // 2
    gatherer_kwargs = {"sample": True}

@train_preference_comparisons_ex.named_config
def cartpole():
    env_name = "CartPole-v1"
    allow_variable_horizon = True


@train_preference_comparisons_ex.named_config
def seals_cartpole():
    env_name = "seals/CartPole-v0"


@train_preference_comparisons_ex.named_config
def pendulum():
    env_name = "Pendulum-v0"


@train_preference_comparisons_ex.named_config
def mountain_car():
    env_name = "MountainCar-v0"
    allow_variable_horizon = True


@train_preference_comparisons_ex.named_config
def seals_mountain_car():
    env_name = "seals/MountainCar-v0"

@train_preference_comparisons_ex.named_config
def shaped_mountain_car():
    env_name = "imitation/MountainCar-v0"
    agent_path = "results/expert_demos/seals_mountain_car/policies/final"
    total_timesteps = int(1e5)
    fragment_length = 20
    total_comparisons = 5000
    comparisons_per_iteration = 300

@train_preference_comparisons_ex.named_config
def sparse_reacher():
    env_name = "imitation/SparseReacher-v0"
    fragment_length = 20


@train_preference_comparisons_ex.named_config
def empty_maze_10():
    env_name = "imitation/EmptyMaze10-v0"
    fragment_length = 4
    random_frac = 0.5
    total_timesteps = int(5e5)
    total_comparisons = 5000
    comparisons_per_iteration = 300
    normalize = False

@train_preference_comparisons_ex.named_config
def empty_maze_4():
    env_name = "imitation/EmptyMaze4-v0"
    fragment_length = 3
    random_frac = 0.5
    total_timesteps = int(1e5)
    total_comparisons = 1000
    comparisons_per_iteration = 100
    normalize = False

@train_preference_comparisons_ex.named_config
def fast(env_name):
    """Minimize the amount of computation.

    Useful for test cases.
    """
    if env_name == "imitation/MountainCar-v0":
        agent_path = "results/expert_demos/seals_mountain_car_fast/policies/final"
    total_timesteps = 2
    total_comparisons = 3
    comparisons_per_iteration = 2
    parallel = False
    num_vec = 1
    fragment_length = 2
    n_episodes_eval = 1
    agent_kwargs = {"batch_size": 2, "n_steps": 10, "n_epochs": 1}
    reward_trainer_kwargs = {
        "epochs": 1,
    }
