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

    save_rollouts = False  # Save rollouts generated during eval to disk?


@eval_policy_ex.named_config
def render():
    train = dict(num_vec=1, parallel=False)
    render = True


@eval_policy_ex.named_config
def fast():
    train = dict(env_name="CartPole-v1", num_vec=1, parallel=False)
    render = True
    policy_type = "ppo"
    policy_path = "tests/testdata/expert_models/cartpole_0/policies/final/"
    eval_n_timesteps = 1
    eval_n_episodes = None
