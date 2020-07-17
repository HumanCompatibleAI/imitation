import os

import sacred

from imitation.util import util

eval_policy_ex = sacred.Experiment("eval_policy")


@eval_policy_ex.config
def replay_defaults():
    env_name = "CartPole-v1"  # environment to evaluate in
    eval_n_timesteps = int(1e4)  # Min timesteps to evaluate, optional.
    eval_n_episodes = None  # Num episodes to evaluate, optional.
    num_vec = 1  # number of environments in parallel
    parallel = False  # Use SubprocVecEnv (generally faster if num_vec>1)
    max_episode_steps = None  # Set to positive int to limit episode horizons

    render = True  # render to screen
    render_fps = 60  # -1 to render at full speed
    log_root = os.path.join("output", "eval_policy")  # output directory

    policy_type = "ppo"  # class to load policy, see imitation.policies.loader
    policy_path = (
        "tests/data/expert_models/" "cartpole_0/policies/final/"
    )  # serialized policy

    reward_type = None  # Optional: override with reward of this type
    reward_path = None  # Path of serialized reward to load


@eval_policy_ex.config
def logging(log_root, env_name):
    log_dir = os.path.join(
        log_root, env_name.replace("/", "_"), util.make_unique_timestamp()
    )


@eval_policy_ex.named_config
def fast():
    eval_n_timesteps = 1
    eval_n_episodes = None
    max_episode_steps = 1
