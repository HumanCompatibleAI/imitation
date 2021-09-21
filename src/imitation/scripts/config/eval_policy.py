"""Configuration settings for eval_policy, evaluating pre-trained policies."""

import sacred

from imitation.scripts.common import train

eval_policy_ex = sacred.Experiment("eval_policy", ingredients=[train.train_ingredient])


@eval_policy_ex.config
def replay_defaults():
    eval_n_timesteps = int(1e4)  # Min timesteps to evaluate, optional.
    eval_n_episodes = None  # Num episodes to evaluate, optional.

    videos = False  # save video files
    video_kwargs = {}  # arguments to VideoWrapper
    render = False  # render to screen
    render_fps = 60  # -1 to render at full speed

    policy_type = None  # class to load policy, see imitation.policies.loader
    policy_path = None  # path to serialized policy

    reward_type = None  # Optional: override with reward of this type
    reward_path = None  # Path of serialized reward to load

    rollout_save_path = None  # where to save rollouts to -- if None, do not save


@eval_policy_ex.named_config
def render():
    train = dict(num_vec=1, parallel=False)
    render = True


@eval_policy_ex.named_config
def acrobot():
    train = dict(env_name="Acrobot-v1")


@eval_policy_ex.named_config
def ant():
    train = dict(env_name="Ant-v2")


@eval_policy_ex.named_config
def cartpole():
    train = dict(env_name="CartPole-v1")


@eval_policy_ex.named_config
def seals_cartpole():
    train = dict(env_name="seals/CartPole-v0")


@eval_policy_ex.named_config
def half_cheetah():
    train = dict(env_name="HalfCheetah-v2")


@eval_policy_ex.named_config
def seals_hopper():
    train = dict(env_name="seals/Hopper-v0")


@eval_policy_ex.named_config
def seals_humanoid():
    train = dict(env_name="seals/Humanoid-v0")


@eval_policy_ex.named_config
def mountain_car():
    train = dict(env_name="MountainCar-v0")


@eval_policy_ex.named_config
def seals_mountain_car():
    train = dict(env_name="seals/MountainCar-v0")


@eval_policy_ex.named_config
def pendulum():
    train = dict(env_name="Pendulum-v0")


@eval_policy_ex.named_config
def reacher():
    train = dict(env_name="Reacher-v2")


@eval_policy_ex.named_config
def seals_ant():
    train = dict(env_name="seals/Ant-v0")


@eval_policy_ex.named_config
def seals_swimmer():
    train = dict(env_name="seals/Swimmer-v0")


@eval_policy_ex.named_config
def seals_walker():
    train = dict(env_name="seals/Walker2d-v0")


@eval_policy_ex.named_config
def fast():
    train = dict(env_name="CartPole-v1", num_vec=1, parallel=False)
    render = True
    policy_type = "ppo"
    policy_path = "tests/testdata/expert_models/cartpole_0/policies/final/"
    eval_n_timesteps = 1
    eval_n_episodes = None
