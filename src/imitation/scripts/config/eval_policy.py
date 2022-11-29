"""Configuration settings for eval_policy, evaluating pre-trained policies."""

import sacred

from imitation.scripts.ingredients import environment, expert
from imitation.scripts.ingredients import logging as logging_ingredient

eval_policy_ex = sacred.Experiment(
    "eval_policy",
    ingredients=[
        logging_ingredient.logging_ingredient,
        environment.environment_ingredient,
        expert.expert_ingredient,
    ],
)


@eval_policy_ex.config
def replay_defaults():
    eval_n_timesteps = int(1e4)  # Min timesteps to evaluate, optional.
    eval_n_episodes = None  # Num episodes to evaluate, optional.

    videos = False  # save video files
    video_kwargs = {}  # arguments to VideoWrapper
    render = False  # render to screen
    render_fps = 60  # -1 to render at full speed

    reward_type = None  # Optional: override with reward of this type
    reward_path = None  # Path of serialized reward to load

    rollout_save_path = None  # where to save rollouts to -- if None, do not save

    explore_kwargs = (
        None  # kwargs to feed to ExplorationWrapper -- if None, do not wrap
    )


@eval_policy_ex.named_config
def explore_eps_greedy():
    explore_kwargs = dict(switch_prob=1.0, random_prob=0.1)


@eval_policy_ex.named_config
def render():
    environment = dict(num_vec=1, parallel=False)
    render = True


@eval_policy_ex.named_config
def acrobot():
    environment = dict(gym_id="Acrobot-v1")


@eval_policy_ex.named_config
def ant():
    environment = dict(gym_id="Ant-v2")


@eval_policy_ex.named_config
def cartpole():
    environment = dict(gym_id="CartPole-v1")


@eval_policy_ex.named_config
def seals_cartpole():
    environment = dict(gym_id="seals/CartPole-v0")


@eval_policy_ex.named_config
def half_cheetah():
    environment = dict(gym_id="HalfCheetah-v2")


@eval_policy_ex.named_config
def seals_half_cheetah():
    environment = dict(gym_id="seals/HalfCheetah-v0")


@eval_policy_ex.named_config
def seals_hopper():
    environment = dict(gym_id="seals/Hopper-v0")


@eval_policy_ex.named_config
def seals_humanoid():
    environment = dict(gym_id="seals/Humanoid-v0")


@eval_policy_ex.named_config
def mountain_car():
    environment = dict(gym_id="MountainCar-v0")


@eval_policy_ex.named_config
def seals_mountain_car():
    environment = dict(gym_id="seals/MountainCar-v0")


@eval_policy_ex.named_config
def pendulum():
    environment = dict(gym_id="Pendulum-v1")


@eval_policy_ex.named_config
def reacher():
    environment = dict(gym_id="Reacher-v2")


@eval_policy_ex.named_config
def seals_ant():
    environment = dict(gym_id="seals/Ant-v0")


@eval_policy_ex.named_config
def seals_swimmer():
    environment = dict(gym_id="seals/Swimmer-v0")


@eval_policy_ex.named_config
def seals_walker():
    environment = dict(gym_id="seals/Walker2d-v0")


@eval_policy_ex.named_config
def fast():
    environment = dict(gym_id="seals/CartPole-v0", num_vec=1, parallel=False)
    render = True
    eval_n_timesteps = 1
    eval_n_episodes = None
