import collections
import functools
import glob
import os
import pickle
from typing import Dict, List, Optional, Tuple, Union

import gym
import numpy as np
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.policies import BasePolicy
import tensorflow as tf

from . import util  # Relative import needed to prevent cycle with __init__.py

TrajectoryList = List[Dict[str, np.ndarray]]
"""A list of trajectory dicts.

Each dict contains the keys 'act', 'obs', and 'rew'. For details on these
key-value pairs, see the docstring for `generate_trajectories`.
"""

TransitionsTuple = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
"""An tuple of obs-act-obs-rew values.

For details see the docstring for `generate_transitions`.
"""


class RandomPolicy(BasePolicy):
  """Returns random actions."""
  def __init__(self, ob_space: gym.Space, ac_space: gym.Space):
    self.ob_space = ob_space
    self.ac_space = ac_space

  def step(self, obs, state=None, mask=None, deterministic=False):
    actions = []
    for ob in obs:
      assert self.ob_space.contains(ob)
      actions.append(self.ac_space.sample())
    return actions, None, None, None

  def proba_step(self, obs, state=None, mask=None):
    raise NotImplementedError()


def get_action_policy(policy, observation, deterministic=False):
  """Gets an action from a Stable Baselines policy after some processing.

  Specifically, clips actions to the action space associated with `policy` and
  automatically accounts for vectorized environments inputs.

  This code was adapted from Stable Baselines' `BaseRLModel.predict()`.

  Args:
    policy (stable_baselines.common.policies.BasePolicy): The policy.
    observation (np.ndarray): The input to the policy network. Can either
      be a single input with shape `policy.ob_space.shape` or a vectorized
      input with shape `(n_batch,) + policy.ob_space.shape`.
    deterministic (bool): Whether or not to return deterministic actions
      (usually means argmax over policy's action distribution).

  Returns:
    action (np.ndarray): The action output of the policy network. If
        `observation` is not vectorized (has shape `policy.ob_space.shape`
        instead of shape `(n_batch,) + policy.ob_space.shape`) then
        `action` has shape `policy.ac_space.shape`.
        Otherwise, `action` has shape `(n_batch,) + policy.ac_space.shape`.
  """
  observation = np.array(observation)
  vectorized_env = BaseRLModel._is_vectorized_observation(observation,
                                                          policy.ob_space)

  observation = observation.reshape((-1, ) + policy.ob_space.shape)
  actions, _, states, _ = policy.step(observation, deterministic=deterministic)

  clipped_actions = actions
  if isinstance(policy.ac_space, gym.spaces.Box):
    clipped_actions = np.clip(actions, policy.ac_space.low,
                              policy.ac_space.high)

  if not vectorized_env:
    clipped_actions = clipped_actions[0]

  return clipped_actions, states


class _TrajectoryAccumulator:
  """Accumulates trajectories step-by-step.

  Used in `generate_trajectories()` only, for collecting completed trajectories
  while ignoring partially-completed trajectories.
  """

  def __init__(self):
    self.partial_trajectories = collections.defaultdict(list)

  def finish_trajectory(self, idx) -> Dict[str, np.ndarray]:
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
    return out_dict_stacked

  def add_step(self, idx, step_dict: Dict[str, np.ndarray]):
    """Add a single step to the partial trajectory identified by `idx`.

    This could correspond to, e.g., one environment managed by a VecEnv.
    """
    self.partial_trajectories[idx].append(step_dict)


def generate_trajectories(policy, env, *, n_timesteps=None, n_episodes=None,
                          ) -> TrajectoryList:
  """Generate trajectory dictionaries from a policy and an environment.

  Args:
    policy (BasePolicy or BaseRLModel): A stable_baselines policy or RLModel,
        trained on the gym environment.
    env (VecEnv or Env or str): The environment(s) to interact with.
    n_timesteps (int): The minimum number of obs-action-obs-reward tuples to
        collect (may collect more if episodes run too long). Set exactly one of
        `n_timesteps` and `n_episodes`, or this function will error.
    n_episodes (int): The number of episodes to finish before returning
        collected tuples. Tuples from parallel episodes underway when the final
        episode is finished will not be returned.
        Set exactly one of `n_timesteps` and `n_episodes`, or this function will
        error.

  Returns:
    trajectories: List of trajectory dictionaries. Each trajectory dictionary
        `traj` has the following keys and values:
         - traj["obs"] is an observations array with N+1 rows, where N depends
           on the particular trajectory.
         - traj["act"] is an actions array with N rows.
         - traj["rew"] is a reward array with shape (N,).
  """
  env = util.maybe_load_env(env, vectorize=True)
  assert util.is_vec_env(env)

  if isinstance(policy, BaseRLModel):
    get_action = policy.predict
    policy.set_env(env)  # This checks that env and policy are compatible.
  else:
    get_action = functools.partial(get_action_policy, policy)

  # Validate end condition arguments and initialize end conditions.
  if n_timesteps is not None and n_episodes is not None:
    raise ValueError("n_timesteps and n_episodes were both set")
  elif n_timesteps is not None:
    assert n_timesteps > 0
    end_cond = "timesteps"
  elif n_episodes is not None:
    assert n_episodes > 0
    end_cond = "episodes"
    episodes_elapsed = 0
  else:
    raise ValueError("Set at least one of n_timesteps and n_episodes")

  # Implements end-condition logic.
  def rollout_done():
    if end_cond == "timesteps":
      # accidentallyquadratic.tumblr.com
      return sum(len(t["obs"]) - 1 for t in trajectories) >= n_timesteps
    elif end_cond == "episodes":
      return len(trajectories) >= n_episodes
    else:
      raise RuntimeError(end_cond)

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
    act_batch, _ = get_action(obs_old_batch)
    obs_batch, rew_batch, done_batch, _ = env.step(act_batch)

    # Track episode count.
    if end_cond == "episodes":
      episodes_elapsed += np.sum(done_batch)

    # Don't save tuples if there is a done. The new_obs for any environment
    # is incorrect for any timestep where there is an episode end.
    # (See GH Issue #1).
    zip_iter = enumerate(
        zip(obs_old_batch, act_batch, obs_batch, rew_batch, done_batch))
    for env_idx, (obs_old, act, obs, rew, done) in zip_iter:
      if done:
        # finish env_idx-th trajectory
        # FIXME: this will break horribly if a trajectory ends after the first
        # action, b/c the trajectory will consist of just a single obs. The
        # "correct" fix for this is to PATCH STABLE BASELINES SO THAT ITS
        # VECENV GIVES US A CORRECT FINAL OBSERVATION TO ADD (see
        # bug #1 in our repo).
        new_traj = trajectories_accum.finish_trajectory(env_idx)
        if not ({'act', 'obs', 'rew'} <= new_traj.keys()):
          raise ValueError("Trajectory does not have expected act/obs/rew "
                           "keys; it probably ended on first step. You should "
                           "PATCH STABLE BASELINES (see bug #1 in our repo).")
        trajectories.append(new_traj)
        trajectories_accum.add_step(env_idx, dict(obs=obs))
        continue
      trajectories_accum.add_step(
          env_idx,
          dict(
              act=act,
              rew=rew,
              # this is in fact not the obs corresponding to `act`, but rather
              # the obs *after* `act` (see above)
              obs=obs))

  # Note that we just drop partial trajectories. This is not ideal for some
  # algos; e.g. BC can probably benefit from partial trajectories, too.

  # Sanity checks.
  for trajectory in trajectories:
    n_steps = len(trajectory["act"])
    # extra 1 for the end
    exp_obs = (n_steps + 1, ) + env.observation_space.shape
    real_obs = trajectory["obs"].shape
    assert real_obs == exp_obs, f"expected shape {exp_obs}, got {real_obs}"
    exp_act = (n_steps, ) + env.action_space.shape
    real_act = trajectory["act"].shape
    assert real_act == exp_act, f"expected shape {exp_act}, got {real_act}"
    exp_rew = (n_steps,)
    real_rew = trajectory["rew"].shape
    assert real_rew == exp_rew, f"expected shape {exp_rew}, got {real_rew}"

  return trajectories


def rollout_stats(policy, env, **kwargs):
  """Rolls out trajectories under the policy and returns various statistics.

  Args:
      policy (stable_baselines.BasePolicy): A stable_baselines Model,
          trained on the gym environment.
      env (VecEnv or Env or str): The environment(s) to interact with.
      n_timesteps (int): The number of rewards to collect.
      n_episodes (int): The number of episodes to finish before we stop
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
    "return": np.asarray([sum(t["rew"]) for t in trajectories]),
    "len": np.asarray([len(t["rew"]) for t in trajectories]),
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


def flatten_trajectories(trajectories: TrajectoryList) -> TransitionsTuple:
  """Flatten a series of trajectory dictionaries into arrays.

  Returns observations, actions, next observations, rewards.

  Args:
      trajectories ([dict]): list of dictionaries returned by `generate`, each
        representing a trajectory and each with "obs", "rew", and "act" keys.

  Returns:
      obs_old (array): A numpy array with shape
          `[n_timesteps] + env.observation_space.shape`. The ith observation in
          this array is the observation seen with the agent chooses action
          `rollout_act[i]`.
      act (array): A numpy array with shape
          `[n_timesteps] + env.action_space.shape`.
      obs_new (array): A numpy array with shape
          `[n_timesteps] + env.observation_space.shape`. The ith observation in
          this array is from the transition state after the agent chooses action
          `rollout_act[i]`.
      rew (array): A numpy array with shape `[n_timesteps]`. The
          reward received on the ith timestep is `rew[i]`.
  """
  keys = ["obs_old", "obs_new", "act", "rew"]
  parts = {key: [] for key in keys}
  for traj_dict in trajectories:
    parts["act"].append(traj_dict["act"])
    parts["rew"].append(traj_dict["rew"])
    obs = traj_dict["obs"]
    parts["obs_old"].append(obs[:-1])
    parts["obs_new"].append(obs[1:])
  cat_parts = {
    key: np.concatenate(part_list, axis=0)
    for key, part_list in parts.items()
  }
  lengths = set(map(len, cat_parts.values()))
  assert len(lengths) == 1, f"expected one length, got {lengths}"
  return cat_parts["obs_old"], cat_parts["act"], cat_parts["obs_new"], \
      cat_parts["rew"]


def generate_transitions(policy, env, *, n_timesteps=None, n_episodes=None,
                         truncate=True) -> TransitionsTuple:
  """Generate old_obs-action-new_obs-reward tuples.

  Args:
    policy (BasePolicy or BaseRLModel): A stable_baselines policy or RLModel,
        trained on the gym environment.
    env (VecEnv or Env or str): The environment(s) to interact with.
    n_timesteps (int): The minimum number of obs-action-obs-reward tuples to
        collect (may collect more if episodes run too long). Set exactly one of
        `n_timesteps` and `n_episodes`, or this function will error.
    n_episodes (int): The number of episodes to finish before returning
        collected tuples. Tuples from parallel episodes underway when the final
        episode is finished will not be returned.
        Set exactly one of `n_timesteps` and `n_episodes`, or this function will
        error.
    truncate (bool): If True and n_timesteps is not None, then drop any
        additional samples to ensure that exactly `n_timesteps` samples are
        returned.
  Returns:
    rollout_obs_old (array): A numpy array with shape
        `[n_samples] + env.observation_space.shape`. The ith observation in
        this array is the observation seen with the agent chooses action
        `rollout_act[i]`. `n_samples` is guaranteed to be at least
        `n_timesteps`, if `n_timesteps` was provided.
    rollout_act (array): A numpy array with shape
        `[n_samples] + env.action_space.shape`.
    rollout_obs_new (array): A numpy array with shape
        `[n_samples] + env.observation_space.shape`. The ith observation in
        this array is from the transition state after the agent chooses action
        `rollout_act[i]`.
    rollout_rewards (array): A numpy array with shape `[n_samples]`. The
        reward received on the ith timestep is `rollout_rewards[i]`.
  """
  traj = generate_trajectories(policy, env, n_timesteps=n_timesteps,
                               n_episodes=n_episodes)
  rollout_arrays = flatten_trajectories(traj)
  if truncate and n_timesteps is not None:
    rollout_arrays = tuple(arr[:n_timesteps] for arr in rollout_arrays)
  return rollout_arrays


def generate_transitions_multiple(policies, env, n_timesteps, *, truncate=True,
                                  ) -> TransitionsTuple:
  """Generate obs-act-obs-rew arrays from several policies.

  Splits the desired number of timesteps evenly between all the policies given.

  Args:
      policies (BasePolicy or [BasePolicy]): A policy
          or a list of policies that will be used to generate
          obs-action-obs triples.

          WARNING:
          Due to the way VecEnvs handle
          episode completion states, the last obs-state-obs triple in every
          episode is omitted. (See GitHub issue #1)
      env (gym.Env): The environment the policy should act in.
      n_timesteps (int): The minimum number of obs-action-obs-reward tuples to
          collect (may collect more if episodes run too long). Set exactly one
          of `n_timesteps` and `n_episodes`, or this function will error.
          If the number of policies given doesn't divide this number evenly,
          then the last policy generates more timesteps than the other policies.
      truncate (bool): If True and n_timesteps is not None, then drop any
          additional samples to ensure that exactly `n_timesteps` samples are
          returned.

  Returns:
      rollout_obs_old (array): A numpy array with shape
          `[n_samples] + env.observation_space.shape`. The ith observation in
          this array is the observation seen with the agent chooses action
          `rollout_act[i]`. `n_samples` is guaranteed to be at least
          `n_timesteps`.
      rollout_act (array): A numpy array with shape
          `[n_samples] + env.action_space.shape`.
      rollout_obs_new (array): A numpy array with shape
          `[n_samples] + env.observation_space.shape`. The ith observation in
          this array is from the transition state after the agent chooses
          action `rollout_act[i]`.
      rollout_rew (array): A numpy array with shape `[n_samples]`. The
          reward received on the ith timestep is `rew[i]`.
  """
  try:
    policies = list(policies)
  except TypeError:
    policies = [policies]

  n_policies = len(policies)
  quot, rem = n_timesteps // n_policies, n_timesteps % n_policies
  tf.logging.debug("rollout.generate_transitions_multiple: quot={}, rem={}"
                   .format(quot, rem))

  obs_old, act, obs_new = [], [], []
  for i, pol in enumerate(policies):
    n_timesteps_ = quot
    if i == n_policies - 1:
      # The final policy also generates the remainder if
      # n_policies doesn't evenly divide n_timesteps.
      n_timesteps_ += rem

    obs_old_, act_, obs_new_, _ = generate_transitions(
        pol, env, n_timesteps=n_timesteps_)
    assert len(obs_new_) == len(act_), (len(obs_new_), len(act_))
    assert len(obs_old_) == len(act_), (len(obs_old_), len(act_))

    if truncate:
      # truncate to get exactly n_timesteps_
      act_ = act_[:n_timesteps_]
      obs_new_ = (obs_new_[:n_timesteps_])
      obs_old_ = obs_old_[:n_timesteps_]

    act.extend(act_)
    obs_new.extend(obs_new_)
    obs_old.extend(obs_old_)

  assert len(obs_old) == len(obs_new)
  assert len(act) >= n_timesteps, (len(act), n_timesteps)
  if truncate:
    assert len(act) == n_timesteps

  return tuple(np.array(x) for x in (obs_old, act, obs_new))


def save(rollout_dir: str,
         policy: BaseRLModel,
         env_name: str,
         step: Union[str, int],
         **kwargs,
         ) -> None:
    """Generate policy rollouts and save them to a pickled TrajectoryList.

    Args:
        rollout_dir: Path to the save directory.
        policy: The stable baselines policy.
        env_name: The environment name.
        step: Either the integer training step or "final" to mark that training
            is finished. Used as a suffix in the save file's basename.
        n_timesteps (Optional[int]): `n_timesteps` argument from
            `generate_trajectories`.
        n_episodes (Optional[int]): `n_episodes` argument from
            `generate_trajectories`.
        truncate (bool): `truncate` argument from `generate_trajectories`.
    """
    filename = util.dump_prefix(policy.__class__, env_name, step) + ".pkl"
    path = os.path.join(rollout_dir, filename)
    traj_list = generate_trajectories(policy, policy.get_env(), **kwargs)
    with open(path, "wb") as f:
      pickle.dump(traj_list, f)
    tf.logging.info("Dumped demonstrations to {}.".format(path))


def load_trajectories(rollout_glob: str,
                      max_n_files: Optional[int] = None,
                      ) -> TrajectoryList:
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

  traj_joined = []  # type: TrajectoryList
  for path in ro_paths:
    with open(path, "rb") as f:
      traj = pickle.load(f)  # type: TrajectoryList
      tf.logging.info(f"Loaded rollouts from '{path}'.")
      traj_joined.extend(traj)

  return traj_joined


def load_transitions(rollout_glob: str,
                     max_n_files: Optional[int] = None,
                     ) -> TransitionsTuple:
  """Load transitions from rollout pickles.

  Args:
      rollout_glob: Glob path to rollout pickles.
      max_n_files: If provided, then only load the most recent `max_n_files`
          files, as sorted by modification times.

  Returns:
      old_obs: Old observations.
      acts: Actions.
      new_obs: New observations.
      rew: Rewards.

  Raises:
      ValueError: No files match the glob.
  """
  traj_list = load_trajectories(rollout_glob, max_n_files)
  return flatten_trajectories(traj_list)
