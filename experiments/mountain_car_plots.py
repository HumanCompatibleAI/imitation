"""Heatmaps and reward plotting code for debugging MountainCar."""

import functools
import os
from pathlib import Path
import pickle
from typing import Dict, List, Optional, Sequence, Union

import gym
from matplotlib import pyplot as plt
import numpy as np
from stable_baselines.common.vec_env import DummyVecEnv, VecNormalize

from imitation import util
from imitation.policies.serialize import load_policy
from imitation.rewards.serialize import load_reward
from imitation.util.reward_wrapper import RewardFn

MC_POS_MIN, MC_POS_MAX = -1.2, 0.6
MC_VEL_MIN, MC_VEL_MAX = -0.07, 0.07
MC_GOAL_POS = 0.5
MC_NUM_ACTS = 3


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
  """Combine with `functools.partial` to create an input-normalizing RewardFn.

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
  return functools.partial(_reward_fn_normalize_inputs, reward_fn=reward_fn,
                           vec_normalize=vec_normalize, **kwargs)


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
    env.unwrapped.state = ob
    next_ob = env.step(act)[0]
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
    reward_fn: RewardFn,
    n_pos_step: int = 18,
    n_vel_step: int = 14,
    mark_goal: bool = True,
    gen_trajs: Optional[List[util.rollout.Trajectory]] = None,
    exp_trajs: Optional[List[util.rollout.Trajectory]] = None,
    legend_on: bool = True,
    title: bool = None,
    heatmap: bool = True,
    filter_trans_by_act: bool = True,
) -> plt.Figure:
  """
  Make a MountainCar heatmap of rewards for a particular action. X axis
  is position. Y axis is velocity.

  Args:
    act: The MountainCar action number whose reward we are evaluating.
      Should be 0, 1, or 2.
    reward_fn: Reward function. Should accept unnormalized inputs.
    n_pos_step: The number of squares that the x axis of the heatmap is divided
      into.
    n_vel_step: The number of squares that the y axis of the heatmap is divided
      into.
    gen_trajs: A list of generator trajectories to
      scatterplot on top of the heatmap.
    exp_trajs: A list of exp trajectories to scatterplot on
      top of the heatmap.
    legend_on: Whether to plot the legend.
    title: Custom title.
    heatmap: Whether to plot the heatmap.
    filter_trans_by_act: If True, then filter out transitions from
      `gen_trajs` and `exp_trajs` that don't use action `act` before
      scatterplotting.
  """
  assert 0 <= act < MC_NUM_ACTS

  pos_space = np.linspace(MC_POS_MIN, MC_POS_MAX, n_pos_step, endpoint=True)
  vel_space = np.linspace(MC_VEL_MIN, MC_VEL_MAX, n_vel_step, endpoint=True)

  obs_vec = np.array([[p, v] for p in pos_space for v in vel_space])
  acts_vec = np.array([act] * len(obs_vec))
  next_obs_vec = make_next_obs(obs_vec, acts_vec)
  steps = np.arange(len(acts_vec))

  rew = reward_fn(obs_vec, acts_vec, next_obs_vec, steps)
  rew_matrix = rew.reshape(n_pos_step, n_vel_step)

  def convert_traj_to_coords_filtered(trajs):
    trans = util.rollout.flatten_trajectories(trajs)
    obs = trans.obs
    if filter_trans_by_act:
      obs = obs[trans.acts == act]
    return obs[:, 0], obs[:, 1]

  fig, ax = plt.subplots()
  if heatmap:
    c = ax.pcolor(pos_space, vel_space, rew_matrix)
    fig.colorbar(c, ax=ax)

  if mark_goal:
    ax.axvline(x=MC_GOAL_POS, linestyle='--',
               label=f"goal state (pos={MC_GOAL_POS})")
  if exp_trajs is not None:
    X, Y = convert_traj_to_coords_filtered(exp_trajs)
    ax.scatter(X, Y, marker="o", label="expert samples", alpha=0.2)
  if gen_trajs is not None:
    X, Y = convert_traj_to_coords_filtered(gen_trajs)
    ax.scatter(X, Y, marker="o", c="yellow", label="policy samples",
               alpha=0.2)

  act_names = ["left", "neutral", "right"]
  if title is None:
    title = f"Action {act_names[act]}"
  ax.set_title(title)
  if legend_on:
    ax.legend(loc="center left", bbox_to_anchor=(0, 1.3))

  return fig


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
      exp_trajs: Expert trajectories for scatterplotting.
          Generator trajectories
          are dynamically generated from generator checkpoints.
  """
  checkpoints_dir = Path(checkpoints_dir)
  for checkpoint_dir in sorted(checkpoints_dir.iterdir()):
    vec_normalize_path = checkpoint_dir / "gen_policy" / "vec_normalize.pkl"
    discrim_path = checkpoint_dir / "discrim"
    policy_path = checkpoint_dir / "gen_policy"

    # Automatically loads VecNormalize for policy evaluation.
    # `gen_trajs` contains unnormalized observations.
    gen_trajs = make_policy_rollout(policy_path)

    # Load VecNormalize for use in RewardFn, which doesn't automatically
    # normalize input observations.
    with open(vec_normalize_path, "rb") as f:
      vec_normalize = pickle.load(f)  # type: VecNormalize
    vec_normalize.training = False

    reward_fn_ctx = load_reward("DiscrimNet", discrim_path, venv=None)
    with reward_fn_ctx as reward_fn:
      norm_rew_fn = build_norm_reward_fn(reward_fn=reward_fn,
                                         vec_normalize=vec_normalize)
      for i in range(MC_NUM_ACTS):
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
    colors: Sequence[str] = ("tab:blue", "tab:orange"),
) -> plt.Figure:
  """Plots rewards received by many trajectories from many agents over time.

  Args:
      trajs_dict: Dictionary mapping rollout labels (e.g. "expert" or
        "gen policy") to rollouts associated with those labels.
      reward_fn: Reward function for evaluating rollout rewards.
      colors: Custom colors for plotted rewards.
  """
  fig, ax = plt.subplots()
  for i, (traj_name, traj_list) in enumerate(trajs_dict.items()):
    X = []
    Y = []
    for traj in traj_list:
      T = len(traj.rews)
      X.extend(range(T))
      steps = np.arange(len(traj.acts))
      rews = reward_fn(traj.obs[:-1], traj.acts, traj.obs[1:], steps)
      Y.extend(rews)
    if i < len(colors):
      color = colors[i]
    else:
      color = None
    ax.plot(X, Y, alpha=1.0, c=color, label=traj_name)
  ax.xlabel("timestep")
  ax.ylabel("test reward")
  ax.legend()
  return fig
