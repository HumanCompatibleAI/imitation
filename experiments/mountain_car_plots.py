"""Heatmaps and reward plotting code for debugging MountainCar."""

from functools import partial
import os
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Union

import gym
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize

from imitation import util
from imitation.policies.serialize import load_policy
from imitation.rewards.serialize import load_reward
from imitation.util.reward_wrapper import RewardFn

MC_POS_MIN, MC_POS_MAX = -1.2, 0.6
MC_VEL_MIN, MC_VEL_MAX = -0.07, 0.07
MC_GOAL_POS = 0.5


# Utility for calculating the next observation s'.
# Required for evaluating AIRL reward at arbitrary (s, a) points.
def make_next_obs(obs, acts) -> np.ndarray:
  """Utility for calculating the next observation s'.

  Required for evaluating AIRL reward at arbitrary (s, a) points.
  """
  env = gym.make("MountainCar-v0")
  obs = np.array(obs)
  acts = np.array(acts)
  next_obs = []
  for ob, act in zip(obs, acts):
    assert ob.shape == (2,)
    env.reset()
    env.env.state = ob
    next_ob, *_ = env.step(act)
    next_obs.append(next_ob)
  return np.array(next_obs)


def make_policy_rollout(policy_path, env_name="MountainCar-v0",
                        ) -> List[util.rollout.Trajectory]:
  """Load policy from path and return a list of Trajectories."""
  venv = DummyVecEnv([lambda: gym.make(env_name)])
  with load_policy("ppo2", str(policy_path), venv) as gen_policy:
    trajs = util.rollout.generate_trajectories(
      gen_policy, venv, sample_until=util.rollout.min_episodes(50))
  return list(trajs)


def make_heatmap(
    act: int,
    reward_fn,
    pos_step_size: float = 0.1,
    velocity_step_size: float = 1e-2,
    mark_goal=True,
    gen_trajs=None,
    exp_trajs=None,
    legend_on=True,
    title=None,
    heatmap=True,
    filter_trans_by_act=True,
) -> None:
  """
  Make a MountainCar heatmap of rewards for a particular action. X axis
  is position. Y axis is velocity.

  Args:
    act (int): The MountainCar action number whose reward we are evaluating.
      Should be 0, 1, or 2.
    reward_fn (RewardFn): Reward function. Should accept unnormed inputs
      (obs, acts, next_obs).
    pos_step_size (float): The x axis granularity.
    velocity_step_size (float): The y axis granularity.
    mark_goal (bool): If True then plot a line for goal position.
    gen_trajs (List[Trajectory]): A list of generator trajectories to
      scatterplot on top of the heatmap.
    exp_trajs (List[Trajectory]): A list of exp trajectories to scatterplot on
      top of the heatmap.
    legend_on (bool): Whether to plot the legend.
    title (str): Custom title.
    heatmap (bool): Whether to plot the heatmap.
    filter_trans_by_act (bool): If True, then filter out transitions from
      `gen_trajs` and `exp_trajs` that don't use action `act` before
      scatterplotting.
  """
  assert 0 <= act <= 2

  obs_vec, acts_vec = [], []
  for p in np.arange(MC_POS_MIN, MC_POS_MAX + 1e-5, step=pos_step_size):
    for v in np.arange(MC_VEL_MIN, MC_VEL_MAX + 1e-5, step=velocity_step_size):
      obs_vec.append([p, v])
      acts_vec.append(act)

  obs_vec = np.array(obs_vec)
  acts_vec = np.array(acts_vec)
  next_obs_vec = make_next_obs(obs_vec, acts_vec)

  R = reward_fn(obs_vec, acts_vec, next_obs_vec, None)

  df = pd.DataFrame()
  df["position"] = np.round(obs_vec[:, 0], 5)
  df["velocity"] = np.round(obs_vec[:, 1], 5)
  df["reward"] = R
  df = df.pivot("velocity", "position", "reward")

  def convert_to_rc(pos, vel):
    x = (np.array(pos) - MC_POS_MIN) / pos_step_size + 0.5
    y = (np.array(vel) - MC_VEL_MIN) / velocity_step_size + 0.5
    return x, y

  def convert_to_rc_filtered(trajs):
    trans = util.rollout.flatten_trajectories(trajs)
    obs = trans.obs
    if filter_trans_by_act:
      obs = obs[trans.acts == act]
    pos, vel = obs[:, 0], obs[:, 1]
    return convert_to_rc(pos, vel)

  plt.figure()
  if heatmap:
    sns.heatmap(df)

  if mark_goal:
    goal_r, _ = convert_to_rc(MC_GOAL_POS, 0)
    plt.axvline(x=goal_r, linestyle='--',
                label=f"goal state (pos={MC_GOAL_POS})")
  if exp_trajs is not None:
    X, Y = convert_to_rc_filtered(exp_trajs)
    plt.scatter(X, Y, marker="o", label="expert samples", alpha=0.2)
  if gen_trajs is not None:
    X, Y = convert_to_rc_filtered(gen_trajs)
    plt.scatter(X, Y, marker="o", c="yellow", label="policy samples",
                alpha=0.2)

  act_names = ["left", "neutral", "right"]
  if title is None:
    title = f"Action {act_names[act]}"
  plt.title(title)
  if legend_on:
    plt.legend(loc="center left", bbox_to_anchor=(0, 1.3))


def batch_reward_heatmaps(
    checkpoints_dir: Union[str, Path],
    output_dir: Union[str, Path] = Path("/tmp/default"),
    exp_trajs: Optional[List[util.rollout.Trajectory]] = None,
) -> None:
  """Save mountain car reward heatmaps for every action and every checkpoint.

  Plots with rollout transitions scatterplotted on top are saved in
  "{output_dir}/act_{i}".
  Plots without rollout transitions are saved in
  "{output_dir}/no_rollout_act_{i}".

  Args:
      checkpoints_dir: Path to `checkpoint` directory from AIRL or GAIL output
          directory. Should contain "gen_policy" and "discrim" directories.
      output_dir: Heatmap output directory.
      exp_trajs: List[Trajectory]: Expert trajectories for scatterplotting.
          Generator trajectories
          are dynamically generated from generator checkpoints.
  """
  checkpoints_dir = Path(checkpoints_dir)
  for checkpoint_dir in sorted(checkpoints_dir.iterdir()):
    vec_normalize_path = checkpoint_dir / "gen_policy" / "vec_normalize.pkl"
    discrim_path = checkpoint_dir / "discrim"
    policy_path = checkpoint_dir / "gen_policy"

    with open(vec_normalize_path, "rb") as f:
      vec_normalize = pickle.load(f)  # type: VecNormalize
    vec_normalize.training = False

    gen_trajs = make_policy_rollout(policy_path)

    reward_fn_ctx = load_reward("DiscrimNet", discrim_path, venv="unused")
    with reward_fn_ctx as reward_fn:
      norm_rew_fn = build_norm_reward_fn(reward_fn=reward_fn,
                                         vec_normalize=vec_normalize)
      for i in range(3):
        make_heatmap(act=i, reward_fn=norm_rew_fn, gen_trajs=gen_trajs,
                     exp_trajs=exp_trajs)
        save_path = Path(output_dir, f"act_{i}", checkpoint_dir.name)
        os.makedirs(save_path.parent, exist_ok=True)
        plt.savefig(save_path)

        make_heatmap(act=i, reward_fn=norm_rew_fn)
        save_path = Path(output_dir, f"no_rollout_act_{i}",
                         checkpoint_dir.name)
        os.makedirs(save_path.parent, exist_ok=True)
        plt.savefig(save_path)
        print(f"saved to {save_path}")


def plot_reward_vs_time(
    trajs_dict: Dict[str, List[util.rollout.Trajectory]],
    reward_fn: RewardFn,
    colors=("tab:blue", "tab:orange"),
):
  """Plots rewards received by many trajectories from many agents over time.

  Args:
      trajs_dict: Dictionary mapping rollout labels (e.g. "expert" or
        "gen policy") to rollouts associated with those labels.
      reward_fn: Reward function for evaluating rollout rewards.
  """
  plt.clf()
  for i, (traj_name, traj_list) in enumerate(trajs_dict.items()):
    X = []
    Y = []
    for traj in traj_list:
      T = len(traj.rews)
      X.extend(range(T))
      rews = reward_fn(traj.obs[:-1], traj.acts, traj.obs[1:], None)
      Y.extend(rews)
    if i < len(colors):
      color = colors[i]
    else:
      color = None
    plt.plot(X, Y, alpha=1.0, c=color, label=traj_name)
  plt.xlabel("timestep")
  plt.ylabel("test reward")
  plt.legend()
  plt.show()


def _reward_fn_normalize_inputs(obs: np.ndarray,
                                acts: np.ndarray,
                                next_obs: np.ndarray,
                                steps: Optional[np.ndarray] = None,
                                *,
                                reward_fn: RewardFn,
                                vec_normalize: VecNormalize,
                                norm_reward: bool = True,
                                verbose: bool = True,
                                ) -> np.ndarray:
  """Combine with `partial` to create an input-normalizing RewardFn.

  Args:
    reward_fn: The reward function that normalized inputs are evaluated on.
    vec_normalize: Instance of VecNormalize used to normalize inputs and
     rewards.
    norm_reward: If True, then also normalize reward before returning.
  Returns:
    The possibly normalized reward.
  """
  norm_obs = vec_normalize.norm_obs(obs)
  norm_next_obs = vec_normalize.norm_obs(next_obs)
  rew = reward_fn(norm_obs, acts, norm_next_obs, steps)
  if norm_reward:
    rew = vec_normalize.normalize_reward(rew)
  if verbose:
    print("rew normed min:", rew.min())
    print("rew normed max:", rew.max())
  return rew


def build_norm_reward_fn(*, reward_fn, vec_normalize, **kwargs) -> RewardFn:
  """Reward function that automatically normalizes inputs.

  See _reward_fn_normalize_inputs for argument documentation.
  """
  return partial(_reward_fn_normalize_inputs, reward_fn=reward_fn,
                 vec_normalize=vec_normalize, **kwargs)
