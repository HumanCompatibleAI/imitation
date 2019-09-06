import collections
import functools
import glob
import os
import pickle
from typing import Dict, List, NamedTuple, Optional, Union

import gym
import numpy as np
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.policies import BasePolicy
from stable_baselines.common.vec_env import VecEnv
import tensorflow as tf

from imitation.policies.base import get_action_policy

from . import util  # Relative import needed to prevent cycle with __init__.py


class Trajectory(NamedTuple):
  """A trajectory."""

  act: np.ndarray
  """Actions: shape (trajectory_len, ) + action_shape."""
  obs: np.ndarray
  """Observations: shape (trajectory_len+1, ) + observation_shape."""
  rew: np.ndarray
  """Reward: shape (trajectory_len, )"""


class Transitions(NamedTuple):
  """A batch of transition obs-act-obs-rew transitions."""
  old_obs: np.ndarray
  """Old observations. Shape: (batch_size, ) + observation_shape.

  The i'th observation old_obs[i] in this array is the observation seen by the
  agent when choosing action act[i].
  """
  act: np.ndarray
  """Actions. Shape: (batch_size, ) + action_shape.
  """
  new_obs: np.ndarray
  """New observation. Shape: (batch_size, ) + observation_shape.

  The i'th observation new_obs[i] in this array is the observation after
  the agent has taken action act[i].
  """
  rew: np.ndarray
  """Reward. Shape: (batch_size, ).

  The reward rew[i] at the i'th timestep is received after the agent has
  taken action act[i].
  """
  done: np.ndarray
  """Boolean array indicating episode termination. Shape: (batch_size, ).

  done[i] is true iff new_obs[i] the last observatoin of an episode.
  """


class _TrajectoryAccumulator:
  """Accumulates trajectories step-by-step.

  Used in `generate_trajectories()` only, for collecting completed trajectories
  while ignoring partially-completed trajectories.
  """

  def __init__(self):
    self.partial_trajectories = collections.defaultdict(list)

  def finish_trajectory(self, idx) -> Trajectory:
    """Complete the trajectory labelled with `idx`.

    Return list of completed trajectories popped from
    `self.partial_trajectories`.
    """
    part_dicts = self.partial_trajectories[idx]
    del self.partial_trajectories[idx]
    out_dict_unstacked = collections.defaultdict(list)
    for part_dict in part_dicts:
      for key, array in part_dict.items():
        out_dict_unstacked[key].append(array)
    out_dict_stacked = {
        key: np.stack(arr_list, axis=0)
        for key, arr_list in out_dict_unstacked.items()
    }
    return Trajectory(**out_dict_stacked)

  def add_step(self, idx, step_dict: Dict[str, np.ndarray]):
    """Add a single step to the partial trajectory identified by `idx`.

    This could correspond to, e.g., one environment managed by a VecEnv.
    """
    self.partial_trajectories[idx].append(step_dict)


def _validate_traj_generate_params(n_timesteps, n_episodes):
  if n_timesteps is not None and n_episodes is not None:
    raise ValueError("n_timesteps and n_episodes were both set")
  elif n_timesteps is not None:
    assert n_timesteps > 0
  elif n_episodes is not None:
    assert n_episodes > 0
  else:
    raise ValueError("Set at least one of n_timesteps and n_episodes")


def generate_trajectories(policy, env, *, n_timesteps=None, n_episodes=None,
                          deterministic_policy=False,
                          ) -> List[Trajectory]:
  """Generate trajectory dictionaries from a policy and an environment.

  Args:
    policy (BasePolicy or BaseRLModel): A stable_baselines policy or RLModel,
        trained on the gym environment.
    env (VecEnv or Env or str): The environment(s) to interact with.
    n_timesteps (int): The minimum number of obs-action-obs-reward tuples to
        collect (may collect more if episodes run too long). Set exactly one of
        `n_timesteps` and `n_episodes`, or this function will error.
    n_episodes (int): The minimum number of episodes to finish before returning
        collected tuples. Tuples from parallel episodes underway when the final
        episode is finished will not be returned.
        Set exactly one of `n_timesteps` and `n_episodes`, or this function will
        error.
    deterministic_policy (bool): If True, asks policy to deterministically
        return action. Note the trajectories might still be non-deterministic
        if the environment has non-determinism!


  Returns:
    List of Trajectory tuples.
  """
  env = util.maybe_load_env(env, vectorize=True)
  assert util.is_vec_env(env)

  if isinstance(policy, BaseRLModel):
    get_action = policy.predict
    policy.set_env(env)  # This checks that env and policy are compatible.
  else:
    get_action = functools.partial(get_action_policy, policy)

  # Validate end condition arguments.
  _validate_traj_generate_params(n_timesteps, n_episodes)

  # Implements end-condition logic.
  def rollout_done():
    if n_timesteps is not None:
      assert n_episodes is None
      # accidentallyquadratic.tumblr.com
      return sum(len(t.obs) - 1 for t in trajectories) >= n_timesteps
    else:
      assert n_episodes is not None
      return len(trajectories) >= n_episodes

  # Collect rollout tuples.
  trajectories = []
  # accumulator for incomplete trajectories
  trajectories_accum = _TrajectoryAccumulator()
  obs_batch = env.reset()
  for env_idx, obs in enumerate(obs_batch):
    # Seed with first obs only. Inside loop, we'll only add second obs from
    # each (s,a,r,s') tuple, under the same "obs" key again. That way we still
    # get all observations, but they're not duplicated into "next obs" and
    # "previous obs" (this matters for, e.g., Atari, where observations are
    # really big).
    trajectories_accum.add_step(env_idx, dict(obs=obs))
  while not rollout_done():
    obs_old_batch = obs_batch
    act_batch, _ = get_action(obs_old_batch, deterministic=deterministic_policy)
    obs_batch, rew_batch, done_batch, info_batch = env.step(act_batch)

    # Don't save tuples if there is a done. The new_obs for any environment
    # is incorrect for any timestep where there is an episode end, so we fix it
    # with returned state info.
    zip_iter = enumerate(
        zip(obs_old_batch, act_batch, obs_batch, rew_batch, done_batch,
            info_batch))
    for env_idx, (obs_old, act, obs, rew, done, info) in zip_iter:
      real_obs = obs
      if done:
        # actual obs is inaccurate, so we use the one inserted into step info
        # by stable baselines wrapper
        real_obs = info['terminal_observation']
      trajectories_accum.add_step(
          env_idx,
          dict(
              act=act,
              rew=rew,
              # this is not the obs corresponding to `act`, but rather the obs
              # *after* `act` (see above)
              obs=real_obs))
      if done:
        # finish env_idx-th trajectory
        new_traj = trajectories_accum.finish_trajectory(env_idx)
        trajectories.append(new_traj)
        trajectories_accum.add_step(env_idx, dict(obs=obs))
        continue

  # Note that we just drop partial trajectories. This is not ideal for some
  # algos; e.g. BC can probably benefit from partial trajectories, too.

  # Sanity checks.
  assert n_episodes is None or len(trajectories) >= n_episodes
  for trajectory in trajectories:
    n_steps = len(trajectory.act)
    # extra 1 for the end
    exp_obs = (n_steps + 1, ) + env.observation_space.shape
    real_obs = trajectory.obs.shape
    assert real_obs == exp_obs, f"expected shape {exp_obs}, got {real_obs}"
    exp_act = (n_steps, ) + env.action_space.shape
    real_act = trajectory.act.shape
    assert real_act == exp_act, f"expected shape {exp_act}, got {real_act}"
    exp_rew = (n_steps,)
    real_rew = trajectory.rew.shape
    assert real_rew == exp_rew, f"expected shape {exp_rew}, got {real_rew}"

  return trajectories


def rollout_stats(policy, env, **kwargs):
  """Rolls out trajectories under the policy and returns various statistics.

  Args:
      policy (stable_baselines.BasePolicy): A stable_baselines Model,
          trained on the gym environment.
      env (VecEnv or Env or str): The environment(s) to interact with.
      n_timesteps (int): The number of rewards to collect.
      n_episodes (int): The minimum number of episodes to finish before we stop
          collecting rewards. Rewards from parallel episodes that are underway
          when the final episode is finished are also included in the return.

  Returns:
      Dictionary containing `n_traj` collected (int), along with return
      statistics (keys: `return_{min,mean,std,max}`, float values) and
      trajectory length statistics (keys: `len_{min,mean,std,max}`, float
      values).
  """
  trajectories = generate_trajectories(policy, env, **kwargs)
  out_stats = {"n_traj": len(trajectories)}
  traj_descriptors = {
    "return": np.asarray([sum(t.rew) for t in trajectories]),
    "len": np.asarray([len(t.rew) for t in trajectories]),
  }
  stat_names = ["min", "mean", "std", "max"]
  for desc_name, desc_vals in traj_descriptors.items():
    for stat_name in stat_names:
      stat_value = getattr(np, stat_name)(desc_vals)
      out_stats[f"{desc_name}_{stat_name}"] = stat_value
  return out_stats


def mean_return(*args, **kwargs) -> float:
  """Find the mean return of a policy.

  Shortcut to call `rollout_stats` and fetch only the value for
  `return_mean`; see documentation for `rollout_stats`.
  """
  return rollout_stats(*args, **kwargs)["return_mean"]


def flatten_trajectories(trajectories: List[Trajectory]) -> Transitions:
  """Flatten a series of trajectory dictionaries into arrays.

  Returns observations, actions, next observations, rewards.

  Args:
      trajectories: list of trajectories.

  Returns:
    The trajectories flattened into a single batch of Transitions.
  """
  keys = ["old_obs", "new_obs", "act", "rew", "done"]
  parts = {key: [] for key in keys}
  for traj in trajectories:
    parts["act"].append(traj.act)
    parts["rew"].append(traj.rew)
    obs = traj.obs
    parts["old_obs"].append(obs[:-1])
    parts["new_obs"].append(obs[1:])
    done = np.zeros_like(traj.rew, dtype=np.bool)
    done[-1] = True
    parts["done"].append(done)
  cat_parts = {
    key: np.concatenate(part_list, axis=0)
    for key, part_list in parts.items()
  }
  lengths = set(map(len, cat_parts.values()))
  assert len(lengths) == 1, f"expected one length, got {lengths}"
  return Transitions(**cat_parts)


def generate_transitions(policy, env, *, n_timesteps=None, n_episodes=None,
                         truncate=True, **kwargs) -> Transitions:
  """Generate old_obs-action-new_obs-reward tuples.

  Args:
    policy (BasePolicy or BaseRLModel): A stable_baselines policy or RLModel,
        trained on the gym environment.
    env (VecEnv or Env or str): The environment(s) to interact with.
    n_timesteps (int): The minimum number of obs-action-obs-reward tuples to
        collect (may collect more if episodes run too long). Set exactly one of
        `n_timesteps` and `n_episodes`, or this function will error.
    n_episodes (int): The minimum number of episodes to finish before returning
        collected tuples. Tuples from parallel episodes underway when the final
        episode is finished will not be returned.
        Set exactly one of `n_timesteps` and `n_episodes`, or this function will
        error.
    truncate (bool): If True and n_timesteps is not None, then drop any
        additional samples to ensure that exactly `n_timesteps` samples are
        returned.
    kwargs (dict): Passed-through to generate_trajectories.
  Returns:
    A batch of Transitions. The length of the constituent arrays is guaranteed
    to be at least `n_timesteps` (if specified), but may be greater unless
    `truncate` is provided as we collect data until the end of each episode.
  """
  traj = generate_trajectories(policy, env, n_timesteps=n_timesteps,
                               n_episodes=n_episodes, **kwargs)
  transitions = flatten_trajectories(traj)
  if truncate and n_timesteps is not None:
    transitions = Transitions(*(arr[:n_timesteps] for arr in transitions))
  return transitions


def save(rollout_dir: str,
         policy: Union[BaseRLModel, BasePolicy],
         env: Union[gym.Env, VecEnv],
         basename: Union[str, int],
         **kwargs,
         ) -> None:
    """Generate policy rollouts and save them to a pickled List[Trajectory].

    Args:
        rollout_dir: Path to the save directory.
        policy: The stable baselines policy.
        env: The environment.
        basename: The file is saved as `f"{basename}.pkl"`. Usually this is
            the step number, or "final".
        n_timesteps (Optional[int]): `n_timesteps` argument from
            `generate_trajectories`.
        n_episodes (Optional[int]): `n_episodes` argument from
            `generate_trajectories`.
        truncate (bool): `truncate` argument from `generate_trajectories`.
    """
    path = os.path.join(rollout_dir, f'{basename}.pkl')
    traj_list = generate_trajectories(policy, env, **kwargs)
    with open(path, "wb") as f:
      pickle.dump(traj_list, f)
    tf.logging.info("Dumped demonstrations to {}.".format(path))


def load_trajectories(rollout_glob: str,
                      max_n_files: Optional[int] = None,
                      ) -> List[Trajectory]:
  """Load trajectories from rollout pickles.

  Args:
      rollout_glob: Glob path to rollout pickles.
      max_n_files: If provided, then only load the most recent `max_n_files`
          files, as sorted by modification times.

  Returns:
      A list of trajectory dictionaries.

  Raises:
      ValueError: No files match the glob.
  """
  ro_paths = glob.glob(rollout_glob)
  if len(ro_paths) == 0:
    raise ValueError(f"No files match glob '{rollout_glob}'")
  if max_n_files is not None:
    ro_paths.sort(key=os.path.getmtime)
    ro_paths = ro_paths[-max_n_files:]

  traj_joined = []  # type: List[Trajectory]
  for path in ro_paths:
    with open(path, "rb") as f:
      traj = pickle.load(f)  # type: List[Trajectory]
      tf.logging.info(f"Loaded rollouts from '{path}'.")
      traj_joined.extend(traj)

  return traj_joined
