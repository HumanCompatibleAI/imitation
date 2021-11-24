"""Configuration for imitation.scripts.train_preference_comparisons."""

import sacred

from imitation.scripts.common import common, reward, rl, train

train_preference_comparisons_ex = sacred.Experiment(
    "train_preference_comparisons",
    ingredients=[
        common.common_ingredient,
        reward.reward_ingredient,
        rl.rl_ingredient,
        train.train_ingredient,
    ],
)


@train_preference_comparisons_ex.config
def train_defaults():
    fragment_length = 50  # timesteps per fragment used for comparisons
    total_timesteps = int(1e6)  # total number of environment timesteps
    total_comparisons = 10000  # total number of comparisons to elicit
    # comparisons to gather before switching back to agent training
    comparisons_per_iteration = 250
    # factor by which to oversample transitions before creating fragments
    transition_oversampling = 1
    # fraction of total_comparisons that will be sampled right at the beginning
    initial_comparison_frac = 0.2
    # fraction of sampled trajectories that will use random actions rather than policy
    random_frac = 0.0

    record_video_every = 0  # record video every N iterations (0 to disable)
    reward_trainer_kwargs = {
        "epochs": 5,
        "weight_decay": 1e-7,
    }
    save_preferences = False  # save preference dataset at the end?
    agent_path = None  # path to a (partially) trained agent to load at the beginning
    gatherer_kwargs = {}
    fragmenter_kwargs = {
        "warning_threshold": 0,
    }
    # path to a pickled sequence of trajectories used instead of training an agent
    trajectory_path = None
    allow_variable_horizon = False

    normalize = True  # Use VecNormalize
    normalize_kwargs = {"norm_reward": False}  # kwargs for `VecNormalize`


@train_preference_comparisons_ex.named_config
def video():
    record_video_every = 1


@train_preference_comparisons_ex.named_config
def cartpole():
    common = dict(env_name="CartPole-v1")
    allow_variable_horizon = True


@train_preference_comparisons_ex.named_config
def seals_cartpole():
    common = dict(env_name="seals/CartPole-v0")


@train_preference_comparisons_ex.named_config
def pendulum():
    common = dict(env_name="Pendulum-v0")


@train_preference_comparisons_ex.named_config
def mountain_car():
    common = dict(env_name="MountainCar-v0")
    allow_variable_horizon = True


@train_preference_comparisons_ex.named_config
def seals_mountain_car():
    common = dict(env_name="seals/MountainCar-v0")


@train_preference_comparisons_ex.named_config
def fast():
    # Minimize the amount of computation. Useful for test cases.
    total_timesteps = 2
    total_comparisons = 3
    comparisons_per_iteration = 2
    fragment_length = 2
    reward_trainer_kwargs = {
        "epochs": 1,
    }
