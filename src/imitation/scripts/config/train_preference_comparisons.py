"""Configuration for imitation.scripts.train_preference_comparisons."""

import sacred
from torch import nn

from imitation.algorithms import preference_comparisons
from imitation.data.wrappers import RenderImageInfoWrapper
from imitation.scripts.ingredients import environment
from imitation.scripts.ingredients import logging as logging_ingredient
from imitation.scripts.ingredients import policy_evaluation, reward, rl

# Note: All the hyperparameter configs in the file are of the tuned
# hyperparameters of the RL algorithm of the respective environment.
# Taken from imitation/scripts/config/train_rl.py

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


@train_preference_comparisons_ex.named_config
def synch_human_preferences():
    gatherer_cls = preference_comparisons.SynchronousHumanGatherer
    gatherer_kwargs = dict(video_dir="videos")
    querent_cls = preference_comparisons.PreferenceQuerent
    querent_kwargs = dict()
    environment = dict(
        post_wrappers=dict(
            RenderImageInfoWrapper=lambda env, env_id, **kwargs: RenderImageInfoWrapper(
                env,
                **kwargs,
            ),
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
        querent_kwargs=dict(
            video_output_dir="../pref_collect/videofiles",
            video_fps=20,
        ),
    )
    environment = dict(
        post_wrappers=dict(
            RenderImageInfoWrapper=lambda env, env_id, **kwargs: RenderImageInfoWrapper(
                env,
                **kwargs,
            ),
        ),
        post_wrappers_kwargs=dict(
            RenderImageInfoWrapper=dict(scale_factor=0.5, use_file_cache=True),
        ),
        env_make_kwargs=dict(render_mode="rgb_array"),
    )


@train_preference_comparisons_ex.named_config
def cartpole():
    environment = dict(gym_id="CartPole-v1")
    allow_variable_horizon = True


@train_preference_comparisons_ex.named_config
def seals_ant():
    environment = dict(gym_id="seals/Ant-v0")
    rl = dict(
        batch_size=2048,
        rl_kwargs=dict(
            batch_size=16,
            clip_range=0.3,
            ent_coef=3.1441389214159857e-06,
            gae_lambda=0.8,
            gamma=0.995,
            learning_rate=0.00017959211641976886,
            max_grad_norm=0.9,
            n_epochs=10,
            # policy_kwargs are same as the defaults
            vf_coef=0.4351450387648799,
        ),
    )


@train_preference_comparisons_ex.named_config
def half_cheetah():
    locals().update(**MUJOCO_SHARED_LOCALS)
    environment = dict(gym_id="HalfCheetah-v2")
    rl = dict(batch_size=16384, rl_kwargs=dict(batch_size=1024))


@train_preference_comparisons_ex.named_config
def seals_half_cheetah():
    environment = dict(gym_id="seals/HalfCheetah-v0")
    rl = dict(
        batch_size=512,
        rl_kwargs=dict(
            batch_size=64,
            clip_range=0.1,
            ent_coef=3.794797423594763e-06,
            gae_lambda=0.95,
            gamma=0.95,
            learning_rate=0.0003286871805949382,
            max_grad_norm=0.8,
            n_epochs=5,
            vf_coef=0.11483689492120866,
        ),
    )
    num_iterations = 50
    total_timesteps = 20000000


@train_preference_comparisons_ex.named_config
def seals_hopper():
    environment = dict(gym_id="seals/Hopper-v0")
    policy = dict(
        policy_cls="MlpPolicy",
        policy_kwargs=dict(
            activation_fn=nn.ReLU,
            net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        ),
    )
    rl = dict(
        batch_size=2048,
        rl_kwargs=dict(
            batch_size=512,
            clip_range=0.1,
            ent_coef=0.0010159833764878474,
            gae_lambda=0.98,
            gamma=0.995,
            learning_rate=0.0003904770450788824,
            max_grad_norm=0.9,
            n_epochs=20,
            vf_coef=0.20315938606555833,
        ),
    )


@train_preference_comparisons_ex.named_config
def seals_swimmer():
    environment = dict(gym_id="seals/Swimmer-v0")
    policy = dict(
        policy_cls="MlpPolicy",
        policy_kwargs=dict(
            activation_fn=nn.ReLU,
            net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        ),
    )
    rl = dict(
        batch_size=2048,
        rl_kwargs=dict(
            batch_size=64,
            clip_range=0.1,
            ent_coef=5.167107294612664e-08,
            gae_lambda=0.95,
            gamma=0.999,
            learning_rate=0.000414936134792374,
            max_grad_norm=2,
            n_epochs=5,
            # policy_kwargs are same as the defaults
            vf_coef=0.6162112311062333,
        ),
    )


@train_preference_comparisons_ex.named_config
def seals_walker():
    environment = dict(gym_id="seals/Walker2d-v0")
    policy = dict(
        policy_cls="MlpPolicy",
        policy_kwargs=dict(
            activation_fn=nn.ReLU,
            net_arch=[dict(pi=[64, 64], vf=[64, 64])],
        ),
    )
    rl = dict(
        batch_size=8192,
        rl_kwargs=dict(
            batch_size=128,
            clip_range=0.4,
            ent_coef=0.00013057334805552262,
            gae_lambda=0.92,
            gamma=0.98,
            learning_rate=0.000138575372312869,
            max_grad_norm=0.6,
            n_epochs=20,
            # policy_kwargs are same as the defaults
            vf_coef=0.6167177795726859,
        ),
    )


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
