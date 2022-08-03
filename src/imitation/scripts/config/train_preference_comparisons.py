"""Configuration for imitation.scripts.train_preference_comparisons."""

import sacred

from imitation.algorithms import preference_comparisons
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


MUJOCO_SHARED_LOCALS = dict(rl=dict(rl_kwargs=dict(ent_coef=0.1)))
ANT_SHARED_LOCALS = dict(
    total_timesteps=int(3e7),
    rl=dict(batch_size=16384),
)


@train_preference_comparisons_ex.config
def train_defaults():
    fragment_length = 100  # timesteps per fragment used for comparisons
    total_timesteps = int(1e6)  # total number of environment timesteps
    total_comparisons = 5000  # total number of comparisons to elicit
    num_iterations = 5  # Arbitrary, should be tuned for the task
    comparison_queue_size = None
    # factor by which to oversample transitions before creating fragments
    transition_oversampling = 1
    # fraction of total_comparisons that will be sampled right at the beginning
    initial_comparison_frac = 0.1
    # fraction of sampled trajectories that will include some random actions
    exploration_frac = 0.0
    cross_entropy_loss_kwargs = {}
    reward_trainer_kwargs = {
        "epochs": 3,
    }
    save_preferences = False  # save preference dataset at the end?
    agent_path = None  # path to a (partially) trained agent to load at the beginning
    # type of PreferenceGatherer to use
    gatherer_cls = preference_comparisons.SyntheticGatherer
    # arguments passed on to the PreferenceGatherer specified by gatherer_cls
    gatherer_kwargs = {}
    fragmenter_kwargs = {
        "warning_threshold": 0,
    }
    # path to a pickled sequence of trajectories used instead of training an agent
    trajectory_path = None
    trajectory_generator_kwargs = {}  # kwargs to pass to trajectory generator
    allow_variable_horizon = False

    checkpoint_interval = 0  # Num epochs between saving (<0 disables, =0 final only)
    query_schedule = "hyperbolic"


@train_preference_comparisons_ex.named_config
def cartpole():
    common = dict(env_name="CartPole-v1")
    allow_variable_horizon = True


@train_preference_comparisons_ex.named_config
def seals_ant():
    locals().update(**MUJOCO_SHARED_LOCALS)
    locals().update(**ANT_SHARED_LOCALS)
    common = dict(env_name="seals/Ant-v0")


@train_preference_comparisons_ex.named_config
def half_cheetah():
    locals().update(**MUJOCO_SHARED_LOCALS)
    common = dict(env_name="HalfCheetah-v2")
    rl = dict(batch_size=16384, rl_kwargs=dict(batch_size=1024))


@train_preference_comparisons_ex.named_config
def seals_hopper():
    locals().update(**MUJOCO_SHARED_LOCALS)
    common = dict(env_name="seals/Hopper-v0")


@train_preference_comparisons_ex.named_config
def seals_humanoid():
    locals().update(**MUJOCO_SHARED_LOCALS)
    common = dict(env_name="seals/Humanoid-v0")
    total_timesteps = int(4e6)


@train_preference_comparisons_ex.named_config
def seals_cartpole():
    common = dict(env_name="seals/CartPole-v0")


@train_preference_comparisons_ex.named_config
def pendulum():
    common = dict(env_name="Pendulum-v1")


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
    total_timesteps = 50
    total_comparisons = 5
    initial_comparison_frac = 0.2
    num_iterations = 1
    fragment_length = 2
    reward_trainer_kwargs = {
        "epochs": 1,
    }
