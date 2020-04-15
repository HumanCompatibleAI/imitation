import collections
import functools
import os
import pickle
from typing import (Callable, Dict, Hashable, List, NamedTuple, Optional,
                    Sequence, Union)

import gym
import numpy as np
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.policies import BasePolicy
from stable_baselines.common.vec_env import VecEnv
import tensorflow as tf

from imitation.policies.base import get_action_policy
from imitation.rewards import common


class Trajectory(NamedTuple):
  """A trajectory, e.g. a one episode rollout from an expert policy.

   Attributes:
    acts: Actions, shape (trajectory_len, ) + action_shape.
    obs: Observations, shape (trajectory_len+1, ) + observation_shape.
    rews: Reward, shape (trajectory_len, ).
    infos: A list of info dicts, length (trajectory_len, ).
  """
  acts: np.ndarray
  obs: np.ndarray
  rews: np.ndarray
  infos: Optional[List[dict]]


class RolloutInfoWrapper(gym.Wrapper):
  """Add the entire episode's rewards and observations to `info` at episode end.

  Whenever done=True, `info["rollouts"]` is a dict with keys
  "obs" and "rews", whose corresponding values hold the Numpy arrays containing
  the raw observations and rewards seen during this episode.
  """
  def __init__(self, env):
    super().__init__(env)
    self._obs = None
    self._rews = None

  def reset(self, **kwargs):
    new_obs = super().reset()
    self._obs = [new_obs]
    self._rews = []
    return new_obs

  def step(self, action):
    obs, rew, done, info = self.env.step(action)
    self._obs.append(obs)
    self._rews.append(rew)

    if done:
      assert "rollout" not in info
      info["rollout"] = {
        "obs": np.stack(self._obs),
        "rews": np.stack(self._rews),
      }
    return obs, rew, done, info


def unwrap_traj(traj: Trajectory) -> Trajectory:
  """Uses `MonitorPlus`-captured `obs` and `rews` to replace fields.

  This can be useful for bypassing other wrappers to retrieve the original
  `obs` and `rews`.

  Fails if `infos` is None or if the Trajectory was generated from an
  environment without imitation.util.MonitorPlus.

  Args:
    traj: A Trajectory generated from `MonitorPlus`-wrapped Environments.

  Returns:
    A copy of `traj` with replaced `obs` and `rews` fields.
  """
  ep_info = traj.infos[-1]["rollout"]
  res = traj._replace(obs=ep_info["obs"], rews=ep_info["rews"])
  assert len(res.obs) == len(res.acts) + 1
  assert len(res.rews) == len(res.acts)
  return res


def recalc_rewards_traj(traj: Trajectory, reward_fn: common.RewardFn,
                        ) -> np.ndarray:
  """Returns the rewards of the trajectory calculated under a diff reward fn."""
  steps = np.arange(len(traj.rews))
  return reward_fn(traj.obs[:-1], traj.acts, traj.obs[1:], steps)


class Transitions(NamedTuple):
  """A batch of obs-act-obs-rew-done transitions.

  Usually generated by combining and processing several Trajectories via
  `flatten_trajectories()`.

  Attributes:
    obs: Previous observations. Shape: (batch_size, ) + observation_shape.
        The i'th observation `obs[i]` in this array is the observation seen
        by the agent when choosing action `act[i]`.
    act: Actions. Shape: (batch_size, ) + action_shape.
    next_obs: New observation. Shape: (batch_size, ) + observation_shape.
        The i'th observation `next_obs[i]` in this array is the observation
        after the agent has taken action `act[i]`.
    rew: Reward. Shape: (batch_size, ).
        The reward `rew[i]` at the i'th timestep is received after the agent has
        taken action `act[i]`.
    done: Boolean array indicating episode termination. Shape: (batch_size, ).
        `done[i]` is true iff `next_obs[i]` the last observation of an episode.
  """

  obs: np.ndarray
  acts: np.ndarray
  next_obs: np.ndarray
  rews: np.ndarray
  dones: np.ndarray


class TrajectoryAccumulator:
  """Accumulates trajectories step-by-step.

  Useful for collecting completed trajectories while ignoring
  partially-completed trajectories (e.g. when rolling out a VecEnv to collect a
  set number of transitions). Each in-progress trajectory is identified by a
  'key', which enables several independent trajectories to be collected at
  once. They key can also be left at its default value of `None` if you only
  wish to collect one trajectory."""

  def __init__(self):
    """Initialise the trajectory accumulator."""
    self.partial_trajectories = collections.defaultdict(list)

  def add_step(self, step_dict: Dict[str, np.ndarray], key: Hashable = None):
    """Add a single step to the partial trajectory identified by `key`.

    Generally a single step could correspond to, e.g., one environment managed
    by a VecEnv.

    Args:
        step_dict: dictionary containing information for the current step. Its
            keys could include any (or all) attributes of a `Trajectory` (e.g.
            "obs", "acts", etc.).
        key: key to uniquely identify the trajectory to append to, if working
            with multiple partial trajectories."""
    self.partial_trajectories[key].append(step_dict)

  def finish_trajectory(self, key: Hashable = None) -> Trajectory:
    """Complete the trajectory labelled with `key`.

    Args:
        key: key uniquely identifying which in-progress trajectory to remove.

    Returns:
        traj: list of completed trajectories popped from
            `self.partial_trajectories`."""
    part_dicts = self.partial_trajectories[key]
    del self.partial_trajectories[key]
    out_dict_unstacked = collections.defaultdict(list)
    for part_dict in part_dicts:
      for key, array in part_dict.items():
        out_dict_unstacked[key].append(array)
    out_dict_stacked = {
      key: np.stack(arr_list, axis=0)
      for key, arr_list in out_dict_unstacked.items()
    }
    traj = Trajectory(**out_dict_stacked)
    assert traj.rews.shape[0] == traj.acts.shape[0] == traj.obs.shape[0] - 1
    return traj

  def add_steps_and_auto_finish(self,
                                acts: np.ndarray,
                                obs: np.ndarray,
                                rews: np.ndarray,
                                dones: np.ndarray,
                                infos: List[dict]) -> List[Trajectory]:
    """Calls `add_step` repeatedly using acts and the returns from `venv.step`.

    Also automatically calls `finish_trajectory()` for each `done == True`.
    Before calling this method, each environment index key needs to be
    initialized with the initial observation (usually from `venv.reset()`).

    See the body of `util.rollout.generate_trajectory` for an example.

    Args:
        acts: Actions passed into `VecEnv.step()`.
        obs: Return value from `VecEnv.step(acts)`.
        rews: Return value from `VecEnv.step(acts)`.
        dones: Return value from `VecEnv.step(acts)`.
        infos: Return value from `VecEnv.step(acts)`.
    Returns:
        A list of completed trajectories. There should be one Trajectory for
        each `True` in the `dones` argument.
    """
    trajs = []
    for env_idx in range(len(obs)):
      assert env_idx in self.partial_trajectories
      assert list(self.partial_trajectories[env_idx][0].keys()) == ["obs"], (
        "Need to first initialize partial trajectory using "
        "self._traj_accum.add_step({'obs': ob}, key=env_idx)")

    zip_iter = enumerate(zip(acts, obs, rews, dones, infos))
    for env_idx, (act, ob, rew, done, info) in zip_iter:
      if done:
        # actual obs is inaccurate, so we use the one inserted into step info
        # by stable baselines wrapper
        real_ob = info['terminal_observation']
      else:
        real_ob = ob

      self.add_step(
        dict(
          acts=act,
          rews=rew,
          # this is not the obs corresponding to `act`, but rather the obs
          # *after* `act` (see above)
          obs=real_ob,
          infos=info),
        env_idx)
      if done:
        # finish env_idx-th trajectory
        new_traj = self.finish_trajectory(env_idx)
        trajs.append(new_traj)
        self.add_step(dict(obs=ob), env_idx)
    return trajs


GenTrajTerminationFn = Callable[[Sequence[Trajectory]], bool]


def min_episodes(n: int) -> GenTrajTerminationFn:
  """Terminate after collecting n episodes of data.

  Argument:
    n: Minimum number of episodes of data to collect.
      May overshoot if two episodes complete simultaneously (unlikely).

  Returns:
    A function implementing this termination condition.
  """
  assert n >= 1
  return lambda trajectories: len(trajectories) >= n


def min_timesteps(n: int) -> GenTrajTerminationFn:
  """Terminate at the first episode after collecting n timesteps of data.

  Arguments:
    n: Minimum number of timesteps of data to collect.
      May overshoot to nearest episode boundary.

  Returns:
    A function implementing this termination condition.
  """
  assert n >= 1

  def f(trajectories: Sequence[Trajectory]):
    timesteps = sum(len(t.obs) - 1 for t in trajectories)
    return timesteps >= n
  return f


def make_sample_until(n_timesteps: Optional[int],
                      n_episodes: Optional[int],
                      ) -> GenTrajTerminationFn:
  """Returns a termination condition sampling until n_timesteps or n_episodes.

  Arguments:
    n_timesteps: Minimum number of timesteps to sample.
    n_episodes: Number of episodes to sample.

  Returns:
    A termination condition.

  Raises:
    ValueError if both or neither of n_timesteps and n_episodes are set,
    or if either are non-positive.
  """
  if n_timesteps is not None and n_episodes is not None:
    raise ValueError("n_timesteps and n_episodes were both set")
  elif n_timesteps is not None:
    assert n_timesteps > 0
    return min_timesteps(n_timesteps)
  elif n_episodes is not None:
    assert n_episodes > 0
    return min_episodes(n_episodes)
  else:
    raise ValueError("Set at least one of n_timesteps and n_episodes")


def generate_trajectories(policy,
                          venv: VecEnv,
                          sample_until: GenTrajTerminationFn,
                          *,
                          deterministic_policy: bool = False,
                          ) -> Sequence[Trajectory]:
  """Generate trajectory dictionaries from a policy and an environment.

  Args:
    policy (BasePolicy or BaseRLModel): A stable_baselines policy or RLModel,
        trained on the gym environment.
    venv: The vectorized environments to interact with.
    sample_until: A function determining the termination condition.
        It takes a sequence of trajectories, and returns a bool.
        Most users will want to use one of `min_episodes` or `min_timesteps`.
    deterministic_policy: If True, asks policy to deterministically return
        action. Note the trajectories might still be non-deterministic if the
        environment has non-determinism!

  Returns:
    Sequence of `Trajectory` named tuples.
  """
  if isinstance(policy, BaseRLModel):
    get_action = policy.predict
    policy.set_env(venv)
  else:
    get_action = functools.partial(get_action_policy, policy)

  # Collect rollout tuples.
  trajectories = []
  # accumulator for incomplete trajectories
  trajectories_accum = TrajectoryAccumulator()
  obs = venv.reset()
  for env_idx, ob in enumerate(obs):
    # Seed with first obs only. Inside loop, we'll only add second obs from
    # each (s,a,r,s') tuple, under the same "obs" key again. That way we still
    # get all observations, but they're not duplicated into "next obs" and
    # "previous obs" (this matters for, e.g., Atari, where observations are
    # really big).
    trajectories_accum.add_step(dict(obs=ob), env_idx)

  while not sample_until(trajectories):
    acts, _ = get_action(obs, deterministic=deterministic_policy)
    obs, rews, dones, infos = venv.step(acts)

    new_trajs = trajectories_accum.add_steps_and_auto_finish(
      acts, obs, rews, dones, infos)
    trajectories.extend(new_trajs)

  # Note that we just drop partial trajectories. This is not ideal for some
  # algos; e.g. BC can probably benefit from partial trajectories, too.

  # Sanity checks.
  for trajectory in trajectories:
    n_steps = len(trajectory.acts)
    # extra 1 for the end
    exp_obs = (n_steps + 1, ) + venv.observation_space.shape
    real_obs = trajectory.obs.shape
    assert real_obs == exp_obs, f"expected shape {exp_obs}, got {real_obs}"
    exp_act = (n_steps, ) + venv.action_space.shape
    real_act = trajectory.acts.shape
    assert real_act == exp_act, f"expected shape {exp_act}, got {real_act}"
    exp_rew = (n_steps,)
    real_rew = trajectory.rews.shape
    assert real_rew == exp_rew, f"expected shape {exp_rew}, got {real_rew}"

  return trajectories


def rollout_stats(trajectories: Sequence[Trajectory]) -> dict:
  """Calculates various stats for a sequence of trajectories.

  Args:
      trajectories: Sequence of `Trajectory`.

  Returns:
      Dictionary containing `n_traj` collected (int), along with episode return
      statistics (keys: `{monitor_,}return_{min,mean,std,max}`, float values)
      and trajectory length statistics (keys: `len_{min,mean,std,max}`, float
      values).

      `return_*` values are calculated from environment rewards.
      `monitor_*` values are calculated from Monitor-captured rewards, and
      are only included if the `trajectories` contain Monitor infos.
  """
  assert len(trajectories) > 0
  out_stats = {"n_traj": len(trajectories)}
  traj_descriptors = {
    "return": np.asarray([sum(t.rews) for t in trajectories]),
    "len": np.asarray([len(t.rews) for t in trajectories]),
  }

  infos_peek = trajectories[0].infos
  if infos_peek is not None and "episode" in infos_peek[-1]:
    monitor_ep_returns = [t.infos[-1]["episode"]["r"] for t in trajectories]
    traj_descriptors["monitor_return"] = np.asarray(monitor_ep_returns)

  stat_names = ["min", "mean", "std", "max"]
  for desc_name, desc_vals in traj_descriptors.items():
    for stat_name in stat_names:
      stat_value = getattr(np, stat_name)(desc_vals)
      out_stats[f"{desc_name}_{stat_name}"] = stat_value
  return out_stats


def mean_return(*args, **kwargs) -> float:
  """Find the mean return of a policy.

  Shortcut to call `generate_trajectories` and fetch the `rollout_stats` value
  for `'return_mean'`; see documentation for `generate_trajectories` and
  `rollout_stats`.
  """
  trajectories = generate_trajectories(*args, **kwargs)
  return rollout_stats(trajectories)["return_mean"]


def flatten_trajectories(trajectories: Sequence[Trajectory]) -> Transitions:
  """Flatten a series of trajectory dictionaries into arrays.

  Returns observations, actions, next observations, rewards.

  Args:
      trajectories: list of trajectories.

  Returns:
    The trajectories flattened into a single batch of Transitions.
  """
  keys = ["obs", "next_obs", "acts", "rews", "dones"]
  parts = {key: [] for key in keys}
  for traj in trajectories:
    parts["acts"].append(traj.acts)
    parts["rews"].append(traj.rews)
    obs = traj.obs
    parts["obs"].append(obs[:-1])
    parts["next_obs"].append(obs[1:])
    dones = np.zeros_like(traj.rews, dtype=np.bool)
    dones[-1] = True
    parts["dones"].append(dones)
  cat_parts = {
    key: np.concatenate(part_list, axis=0)
    for key, part_list in parts.items()
  }
  lengths = set(map(len, cat_parts.values()))
  assert len(lengths) == 1, f"expected one length, got {lengths}"
  return Transitions(**cat_parts)


def generate_transitions(policy,
                         venv,
                         n_timesteps: int,
                         *,
                         truncate: bool = True,
                         **kwargs) -> Transitions:
  """Generate obs-action-next_obs-reward tuples.

  Args:
    policy (BasePolicy or BaseRLModel): A stable_baselines policy or RLModel,
        trained on the gym environment.
    venv: The vectorized environments to interact with.
    n_timesteps: The minimum number of timesteps to sample.
    truncate: If True, then drop any additional samples to ensure that exactly
        `n_timesteps` samples are returned.
    **kwargs: Passed-through to generate_trajectories.

  Returns:
    A batch of Transitions. The length of the constituent arrays is guaranteed
    to be at least `n_timesteps` (if specified), but may be greater unless
    `truncate` is provided as we collect data until the end of each episode.
  """
  traj = generate_trajectories(policy, venv,
                               sample_until=min_timesteps(n_timesteps),
                               **kwargs)
  transitions = flatten_trajectories(traj)
  if truncate and n_timesteps is not None:
    transitions = Transitions(*(arr[:n_timesteps] for arr in transitions))
  return transitions


def save(path: str,
         policy: Union[BaseRLModel, BasePolicy],
         venv: VecEnv,
         sample_until: GenTrajTerminationFn,
         *,
         unwrap: bool = True,
         exclude_infos: bool = True,
         verbose: bool = True,
         **kwargs,
         ) -> None:
    """Generate policy rollouts and save them to a pickled Sequence[Trajectory].

    The `.infos` field of each Trajectory is set to `None` to save space.

    Args:
      path: Rollouts are saved to this path.
      venv: The vectorized environments.
      sample_until: End condition for rollout sampling.
      unwrap: If True, then save original observations and rewards (instead of
        potentially wrapped observations and rewards) by calling
        `unwrap_traj()`.
      exclude_infos: If True, then exclude `infos` from pickle by setting
        this field to None. Excluding `infos` can save a lot of space during
        pickles.
      verbose: If True, then print out rollout stats before saving.
      deterministic_policy: Argument from `generate_trajectories`.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    trajs = generate_trajectories(policy, venv, sample_until, **kwargs)
    if unwrap:
      trajs = [unwrap_traj(traj) for traj in trajs]
    if exclude_infos:
      trajs = [traj._replace(infos=None) for traj in trajs]
    if verbose:
      stats = rollout_stats(trajs)
      tf.logging.info(f"Rollout stats: {stats}")

    with open(path, "wb") as f:
      pickle.dump(trajs, f)
    tf.logging.info("Dumped demonstrations to {}.".format(path))
