"""Configuration for imitation.scripts.train_preference_comparisons."""

import sacred

from imitation.algorithms import preference_comparisons
from imitation.data.wrappers import RenderImageInfoWrapper
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
    save_preferences = False  # save preference dataset at the end?
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

    # If set, save trajectory videos to this directory. Must be present if gather_cls is
    # SynchronousCLIGatherer
    video_log_dir = None


@train_preference_comparisons_ex.named_config
def synch_human_preferences():
    gatherer_cls = preference_comparisons.SynchronousHumanGatherer
    gatherer_kwargs = dict(
        video_dir="videos"
    )
    querent_cls = preference_comparisons.PreferenceQuerent
    querent_kwargs = dict()
    environment = dict(
        post_wrappers=dict(
            RenderImageInfoWrapper=lambda env, env_id, **kwargs:
                RenderImageInfoWrapper(env, **kwargs),
        ),
        num_vec=2,
        post_wrappers_kwargs=dict(
            RenderImageInfoWrapper=dict(scale_factor=0.5, use_file_cache=True),
        ),
    )


@train_preference_comparisons_ex.named_config
def human_preferences():
    gatherer_cls = preference_comparisons.PrefCollectGatherer
    gatherer_kwargs = dict(
        pref_collect_address="http://127.0.0.1:8000",
        wait_for_user=True,
    )
    querent_cls = preference_comparisons.PrefCollectQuerent
    querent_kwargs = dict(
        pref_collect_address="http://127.0.0.1:8000",
        video_output_dir="../pref-collect/videofiles",
        video_fps=20,
    )
    environment = dict(
        post_wrappers=dict(
            RenderImageInfoWrapper=lambda env, env_id, **kwargs:
                RenderImageInfoWrapper(env, **kwargs),
        ),
        post_wrappers_kwargs=dict(
            RenderImageInfoWrapper=dict(scale_factor=0.5, use_file_cache=True),
        ),
    )


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
