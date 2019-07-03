import functools

import gym
import numpy as np
from stable_baselines.common import BaseRLModel
import tensorflow as tf

from . import util  # Relative import needed to prevent cycle with __init__.py


def get_action_policy(policy, observation, deterministic=False):
  """Get an action from a Stable Baselines policy, while accounting for
  clipping and vectorised environments. This code was adapted from Stable
  Baselines' BaseRLModel.predict()."""
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
  """Accumulates trajectories step-by-step. For use in generate() only."""

  def __init__(self):
    self.partial_trajectories = {}

  def finish_trajectory(self, idx):
    """Complete the trajectory labelled with `idx`, and return list of
    completed trajectories popped from self.partial_trajectories."""
    if idx not in self.partial_trajectories:
      return []
    part_dicts = self.partial_trajectories[idx]
    out_dict_unstacked = {}
    for part_dict in part_dicts:
      for key, array in part_dict.items():
        out_dict_unstacked.setdefault(key, []).append(array)
    final_dict = {
        key: np.stack(arr_list, axis=0)
        for key, arr_list in out_dict_unstacked.items()
    }
    del self.partial_trajectories[idx]
    return [final_dict]

  def add_step(self, idx, step_dict):
    """Add a single step to the partial trajectory identified by `idx` (this
    could correspond to, e.g., one environment managed by a vecenv)."""
    self.partial_trajectories.setdefault(idx, []).append(step_dict)


def generate(policy, env, *, n_timesteps=None, n_episodes=None):
  """
  Generate old_obs-action-new_obs-reward tuples from a policy and an
  environment.

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

  Return:
    rollout_obs_old (array): A numpy array with shape
        `[n_timesteps] + env.observation_space.shape`. The ith observation in
        this array is the observation seen with the agent chooses action
        `rollout_act[i]`.
    rollout_act (array): A numpy array with shape
        `[n_timesteps] + env.action_space.shape`.
    rollout_obs_new (array): A numpy array with shape
        `[n_timesteps] + env.observation_space.shape`. The ith observation in
        this array is from the transition state after the agent chooses action
        `rollout_act[i]`.
    rollout_rewards (array): A numpy array with shape `[n_timesteps]`. The
        reward received on the ith timestep is `rollout_rewards[i]`.
  """
  env = util.maybe_load_env(env, vectorize=True)
  assert util.is_vec_env(env)

  if isinstance(policy, BaseRLModel):
    get_action = policy.predict
    policy.set_env(env)  # This checks that env and policy are compatbile.
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
  for idx, obs in enumerate(obs_batch):
    # Seed with first obs only. Inside loop, we'll only add second obs from
    # each (s,a,r,s') tuple, under the same "obs" key again. That way we still
    # get all observations, but they're not duplicated into "next obs" and
    # "previous obs" (this matters for, e.g., Atari, where observations are
    # really big).
    trajectories_accum.add_step(idx, dict(obs=obs))
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
    for idx, (obs_old, act, obs, rew, done) in zip_iter:
      if done:
        # finish idx-th trajectory
        # FIXME: this will break horribly if a trajectory ends after the first
        # action, b/c the trajectory will consist of just a single obs. The
        # "correct" fix for this is to PATCH STABLE BASELINES SO THAT ITS
        # VECENV GIVES US A CORRECT FINAL OBSERVATION TO ADD.
        trajectories.extend(trajectories_accum.finish_trajectory(idx))
        trajectories_accum.add_step(idx, dict(obs=obs))
        continue
      trajectories_accum.add_step(
          idx,
          dict(
              act=act,
              rew=rew,
              # this is in fact not the obs corresponding to `act`, but rather
              # the obs *after* `act` (see above)
              obs=obs, ))

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
  """Roll out some trajectories under given policy and return a dictionary of
  statistics.

  Args:
      policy (stable_baselines.BasePolicy): A stable_baselines Model,
          trained on the gym environment.
      env (VecEnv or Env or str): The environment(s) to interact with.
      n_timesteps (int): The number of rewards to collect.
      n_episodes (int): The number of episodes to finish before we stop
          collecting rewards. Rewards from parallel episodes that are underway
          when the final episode is finished are also included in the return.

  Return:
      Dictionary containing `n_traj` collected (int), along with return
      statistics (keys: `return_{min,mean,std,max}`, float values) and
      trajectory length statistics (keys: `len_{min,mean,std,max}`, float
      values)."""
  trajectories = generate(policy, env, **kwargs)
  out_stats = {"n_traj": len(trajectories)}
  traj_descriptors = {
    "return": np.asarray([sum(t["rew"]) for t in trajectories]),
    "len": np.asarray([len(t["rew"]) for t in trajectories])
  }
  stat_names = ["min", "mean", "std", "max"]
  for desc_name, desc_vals in traj_descriptors.items():
    for stat_name in stat_names:
      stat_value = getattr(np, stat_name)(desc_vals)
      out_stats[f"{desc_name}_{stat_name}"] = stat_value
  return out_stats


def mean_return(*args, **kwargs):
  """Shortcut to call `rollout_stats` and fetch only the value for
  `return_mean`; see documentation for `rollout_stats`."""
  return rollout_stats(*args, **kwargs)["return_mean"]


def flatten_trajectories(trajectories):
  """Flatten a series of trajectory dictionaries into conjoined sequences of
  observations, actions, next observations, and rewards.

  Args:
      trajectories ([dict]): list of dictionaries returned by `generate`, each
        representing a trajectories and each with "obs", "rew", and "act" keys.

  Returns: series of float arrays, each of same length.
      obs_old (float array): starting state for transition.
      act (float array): action taken during each transition.
      obs_new (float_array): destination state during each transition.
      rew (float array): reward incurred during each transition."""
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


def generate_multiple(policies, env, n_timesteps, *, truncate=True):
  """Generate obs-act-obs triples from several policies.

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
      n_timesteps (int): The number of obs-action-obs
          triples to generate. If the number of policies given doesn't
          divide this number evenly, then the last policy generates
          more timesteps.
      truncate (bool): should collected episodes be truncated so that *only*
          `n_timesteps` are returned? Otherwise this will potentially return
          more timesteps.

  Returns:
      rollout_obs_old (array): A numpy array with shape
          `[n_timesteps] + env.observation_space.shape`. The ith observation in
          this array is the observation seen with the agent chooses action
          `rollout_act[i]`.
      rollout_act (array): A numpy array with shape
          `[n_timesteps] + env.action_space.shape`.
      rollout_obs_new (array): A numpy array with shape
          `[n_timesteps] + env.observation_space.shape`. The ith observation in
          this array is from the transition state after the agent chooses
          action `rollout_act[i]`.
  """
  try:
    policies = list(policies)
  except TypeError:
    policies = [policies]

  n_policies = len(policies)
  quot, rem = n_timesteps // n_policies, n_timesteps % n_policies
  tf.logging.debug("rollout.generate_multiple: quot={}, rem={}"
                   .format(quot, rem))

  obs_old, act, obs_new = [], [], []
  for i, pol in enumerate(policies):
    n_timesteps_ = quot
    if i == n_policies - 1:
      # The final policy also generates the remainder if
      # n_policies doesn't evenly divide n_timesteps.
      n_timesteps_ += rem

    obs_old_, act_, obs_new_, _ = flatten_trajectories(
        generate(pol, env, n_timesteps=n_timesteps_))
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
