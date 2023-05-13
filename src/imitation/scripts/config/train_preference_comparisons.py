"""Configuration for imitation.scripts.train_preference_comparisons."""

import sacred

from imitation.algorithms import preference_comparisons
from imitation.scripts.ingredients import environment
from imitation.scripts.ingredients import logging as logging_ingredient
from imitation.scripts.ingredients import policy_evaluation, reward, rl

train_preference_comparisons_ex = sacred.Experiment(
    "train_preference_comparisons",
    ingredients=[
        logging_ingredient.logging_ingredient,
        environment.environment_ingredient,
        reward.reward_ingredient,
        rl.rl_ingredient,
        policy_evaluation.policy_evaluation_ingredient,
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
    preference_model_kwargs = {}
    reward_trainer_kwargs = {
        "epochs": 3,
    }
    agent_path = None  # path to a (partially) trained agent to load at the beginning
    # type of PreferenceGatherer to use
    gatherer_cls = preference_comparisons.SyntheticGatherer
    # arguments passed on to the PreferenceGatherer specified by gatherer_cls
    gatherer_kwargs = {}
    active_selection = False
    active_selection_oversampling = 2
    uncertainty_on = "logit"
    fragmenter_kwargs = {
        "warning_threshold": 0,
    }
    # path to a pickled sequence of trajectories used instead of training an agent
    trajectory_path = None
    trajectory_generator_kwargs = {}  # kwargs to pass to trajectory generator
    allow_variable_horizon = False

    checkpoint_interval = 0  # Num epochs between saving (<0 disables, =0 final only)
    query_schedule = "hyperbolic"
    save_preferences = False
    bypass_reward_net = False
    initial_epoch_multiplier = 200.0


@train_preference_comparisons_ex.named_config
def cartpole():
    environment = dict(gym_id="CartPole-v1")
    allow_variable_horizon = True


@train_preference_comparisons_ex.named_config
def seals_ant():
    locals().update(**MUJOCO_SHARED_LOCALS)
    locals().update(**ANT_SHARED_LOCALS)
    environment = dict(gym_id="seals/Ant-v0")


@train_preference_comparisons_ex.named_config
def half_cheetah():
    locals().update(**MUJOCO_SHARED_LOCALS)
    environment = dict(gym_id="HalfCheetah-v2")
    rl = dict(batch_size=16384, rl_kwargs=dict(batch_size=1024))


@train_preference_comparisons_ex.named_config
def seals_half_cheetah():
    environment = dict(gym_id="seals/HalfCheetah-v0")
    rl = dict(batch_size=16384, rl_kwargs=dict(batch_size=1024))


@train_preference_comparisons_ex.named_config
def seals_hopper():
    locals().update(**MUJOCO_SHARED_LOCALS)
    environment = dict(gym_id="seals/Hopper-v0")


@train_preference_comparisons_ex.named_config
def seals_humanoid():
    locals().update(**MUJOCO_SHARED_LOCALS)
    environment = dict(gym_id="seals/Humanoid-v0")
    total_timesteps = int(4e6)


@train_preference_comparisons_ex.named_config
def seals_cartpole():
    environment = dict(gym_id="seals/CartPole-v0")


@train_preference_comparisons_ex.named_config
def pendulum():
    environment = dict(gym_id="Pendulum-v1")


@train_preference_comparisons_ex.named_config
def mountain_car():
    environment = dict(gym_id="MountainCar-v0")
    allow_variable_horizon = True


@train_preference_comparisons_ex.named_config
def seals_mountain_car():
    environment = dict(gym_id="seals/MountainCar-v0")


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


@train_preference_comparisons_ex.named_config
def real_reward():
    bypass_reward_net = True
    reward_trainer_kwargs = {
        "epochs": 1,  # 5 might be better? but 1 looks better in his thing
    }


@train_preference_comparisons_ex.named_config
def best_hps():
    initial_comparison_frac = 0.2
    num_iterations = 20
    reward_trainer_kwargs = {
        "epochs": 1,  # 5 might be better? but 1 looks better in his thing
    }
    rl = {
        "batch_size": 512,
        "rl_kwargs": {
            "ent_coef": 0.95,  # this number looks weird
            "n_epochs": 5,
            "gamma": 0.95,
            "gae_lambda": 0.95,
            "clip_range": 0.1,
            "max_grad_norm": 0.8,
            "vf_coef": 0.11483689492120900,
        },
    }
    # reward = {"normalize_output_layer": None}
    total_timesteps = 20000000.0


@train_preference_comparisons_ex.named_config
def more_hps():
    initial_comparison_frac = 0.2
    num_iterations = 20
    reward_trainer_kwargs = {
        "epochs": 1,  # 5 might be better? but 1 looks better in his thing
    }
    rl = {
        "batch_size": 512,
        "rl_kwargs": {
            "ent_coef": 3.992371122209408e-6,
            "n_epochs": 5,
            "gamma": 0.95,
            "gae_lambda": 0.95,
            "clip_range": 0.1,
            "max_grad_norm": 0.8,
            "vf_coef": 0.11483689492120900,
        },
    }
    # reward = {"normalize_output_layer": None}
    total_timesteps = 20000000.0


@train_preference_comparisons_ex.named_config
def initial_epoch_multiplier():
    initial_epoch_multiplier = 1.0
    reward_trainer_kwargs = {"epochs": 5}
