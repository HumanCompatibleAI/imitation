import os

import sacred

from imitation.util import util

eval_policy_ex = sacred.Experiment("eval_policy")


@eval_policy_ex.config
def replay_defaults():
  env_name = "CartPole-v1"  # environment to evaluate in
  timesteps = int(1e4)  # number of timesteps to evaluate
  num_vec = 1  # number of environments in parallel
  parallel = False  # Use SubprocVecEnv (generally faster if num_vec>1)

  render = True  # render to screen
  render_fps = 60  # -1 to render at full speed
  log_root = os.path.join("output", "eval_policy")  # output directory

  policy_type = "ppo2"  # class to load policy, see imitation.policies.loader
  policy_path = "expert_models/PPO2_CartPole-v1_0"  # serialized policy

  reward_type = None  # Optional: override with reward of this type
  reward_path = None  # Path of serialized reward to load


@eval_policy_ex.config
def logging(log_root, env_name):
  log_dir = os.path.join(log_root, env_name.replace("/", "_"),
                         util.make_unique_timestamp())


@eval_policy_ex.named_config
def fast():
  timesteps = int(1e2)
