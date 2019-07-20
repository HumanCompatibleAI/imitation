"""Density-based baselines for imitation learning. Each of these algorithms
learns a density estimate on some aspect of the demonstrations, then rewards
the agent for following that estimate."""

from gym.spaces.utils import flatten
import numpy as np
from sklearn.neighbors import KernelDensity
from sklearn.preprocessing import StandardScaler
import tensorflow as tf

from imitation.util import (FeedForward32Policy, RewardVecEnvWrapper, rollout,
                            util)

# Constants identifying different kinds of density we can use. Note that all
# can be augmented to depend on the time step by passing `is_stationary = True`
# to `DensityReward`.

# Density on state s
STATE_DENSITY = 'state_density'
# Density on (s,a) pairs
STATE_ACTION_DENSITY = 'state_action_density'
# Density (s,s') pairs
STATE_STATE_DENSITY = 'state_state_density'


class DensityReward:
  def __init__(self,
               trajectories,
               density_type,
               kernel,
               obs_space,
               act_space,
               *,
               is_stationary=True,
               standardise=True):
    self.density_type = density_type
    self.is_stationary = is_stationary
    self.kernel = kernel
    self.standardise = standardise
    self.obs_space = obs_space
    self.act_space = act_space
    self._fit_models(trajectories)

  def _fit_models(self, trajectories):
    flat_trajs = self._flatten_trajectories(trajectories)

    # if requested, we'll scale demonstration transitions so that they have
    # zero mean and unit variance (i.e all components are equally important)
    flattened_dataset = np.stack(sum(flat_trajs, []), axis=0)
    self._scaler = StandardScaler(with_mean=self.standardise,
                                  with_std=self.standardise)
    self._scaler.fit(flattened_dataset)

    # now fit density model
    # TODO: add absorbing state fix that I describe in __call__
    if self.is_stationary:
      # fit to all pairs, since density model is stationary
      self._density_model = self._fit_single_density(
          self._scaler.transform(flattened_dataset))
    else:
      # fit separately for samples at each time step
      T = max(map(len, flat_trajs))
      traj_groups = [[]] * T
      for traj in flat_trajs:
        for t, flat_trans in enumerate(traj):
          traj_groups[t].append(flat_trans)
      traj_groups_scaled = [
          self._scaler.transform(np.stack(step_transitions, axis=0))
          for step_transitions in traj_groups
      ]
      self._density_models = [
          self._fit_single_density(scaled_flat_trans)
          for scaled_flat_trans in traj_groups_scaled
      ]

  def _fit_single_density(self, flat_transitions):
    density_model = KernelDensity(kernel=self.kernel)
    density_model.fit(flat_transitions)
    return density_model

  def _flatten_trajectories(self, trajectories):
    flat_trajectories = []
    for traj in trajectories:
      obs_vec = traj['obs']
      act_vec = traj['act']
      assert len(obs_vec) == len(act_vec) + 1
      flat_traj = []
      for step_num in range(len(traj['act'])):
        flat_trans = self._flatten_transition(obs_vec[step_num],
                                              act_vec[step_num],
                                              obs_vec[step_num + 1])
        flat_traj.append(flat_trans)
      flat_trajectories.append(flat_traj)
    return flat_trajectories

  def _flatten_transition(self, obs, act, next_obs):
    if self.density_type == STATE_DENSITY:
      return flatten(self.obs_space, obs)
    elif self.density_type == STATE_ACTION_DENSITY:
      return np.concatenate([
          flatten(self.obs_space, obs),
          flatten(self.act_space, act),
      ])
    elif self.density_type == STATE_STATE_DENSITY:
      return np.concatenate([
          flatten(self.obs_space, obs),
          flatten(self.obs_space, next_obs),
      ])
    else:
      raise ValueError(f"Unknown density type {self.density_type}")

  def __call__(self, obs_b, act_b, next_obs_b, *, steps=None):
    """Compute reward from given (s,a,s') transition batch."""
    rew_list = []
    assert len(obs_b) == len(act_b) and len(obs_b) == len(next_obs_b)
    for idx, (obs, act, next_obs) in enumerate(zip(obs_b, act_b, next_obs_b)):
      flat_trans = self._flatten_transition(obs, act, next_obs)
      scaled_padded_trans = self._scaler.transform(flat_trans[None])
      if self.is_stationary:
        rew = self._density_model.score(scaled_padded_trans)
      else:
        time = steps[idx]
        if time >= len(self._density_models):
          # Can't do anything sensible here yet. Correct solution is to use
          # hierarchical model in which we first check whether state is
          # absorbing, then assign either constant score or a score based on
          # density.
          raise Exception(
              f"Time {time} out of range (0, {len(self._density_models)}], and "
              f"I haven't implemented absorbing states etc. yet")
        else:
          time_model = self._density_models[time]
          rew = time_model.score(scaled_padded_trans)
      rew_list.append(rew)
    return np.asarray(rew_list, dtype='float32')


class DensityTrainer:
  """Family of simple imitation learning baseline algorithms that apply RL to a
  rough density estimate of the demonstration trajectories."""
  def __init__(self,
               env,
               imitation_trainer,
               expert_trainer,
               density_type,
               kernel,
               *,
               is_stationary=True,
               standardise_rew=True,
               policy_class=FeedForward32Policy,
               n_expert_trajectories=5):
    self.env = util.maybe_load_env(env, vectorize=True)
    self.policy_class = policy_class
    self.imitation_trainer = imitation_trainer
    expert_trajectories = rollout.generate_trajectories(
        expert_trainer, self.env, n_episodes=n_expert_trajectories)
    self.reward_fn = DensityReward(trajectories=expert_trajectories,
                                   density_type=density_type,
                                   kernel=kernel,
                                   obs_space=self.env.observation_space,
                                   act_space=self.env.action_space,
                                   is_stationary=is_stationary,
                                   standardise=standardise_rew)
    self.wrapped_env = RewardVecEnvWrapper(self.env,
                                           self.reward_fn,
                                           include_steps=True)
    self.graph = tf.Graph()
    self.sess = tf.Session(graph=self.graph)
    with self.graph.as_default():
      #   self._build_tf_graph()
      self.sess.run(tf.global_variables_initializer())

  def train_policy(self, n_timesteps=int(1e6), **kwargs):
    """Should train for a LONG time (e.g 1e6 steps)."""
    self.imitation_trainer.set_env(self.wrapped_env)
    # FIXME: this is not meant to be called frequently b/c there are
    # significant overheads (see Adam's comment in trainer.py)
    self.imitation_trainer.learn(n_timesteps,
                                 reset_num_timesteps=False,
                                 **kwargs)

  def test_policy(self, *, n_trajectories=10):
    """Test current imitation policy on environment & give some rollout
    stats.

    Args:
      n_trajectories (int): number of rolled-out trajectories.

    Returns:
      dict: rollout statistics collected by
        `imitation.utils.rollout.rollout_stats()`.
    """
    self.imitation_trainer.set_env(self.env)
    reward_stats = rollout.rollout_stats(self.imitation_trainer,
                                         self.env,
                                         n_episodes=n_trajectories)
    return reward_stats
