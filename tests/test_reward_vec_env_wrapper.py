import numpy as np

from imitation import util
from imitation.policies.base import RandomPolicy


class FunkyReward:
  def __call__(self, obs, act, next_obs, *, steps=None):
    # give each environment number from 1 to num_envs
    return (np.arange(len(obs)) + 1).astype('float32')


def test_reward_overwrite():
  """Test that reward wrapper actually overwrites base rewards."""
  env_id = 'Pendulum-v0'
  num_envs = 3
  env = util.make_vec_env(env_id, num_envs)
  reward_fn = FunkyReward()
  wrapped_env = util.reward_wrapper.RewardVecEnvWrapper(env, reward_fn)
  policy = RandomPolicy(env.observation_space, env.action_space)
  default_stats = util.rollout.rollout_stats(policy, env, n_episodes=10)
  wrapped_stats = util.rollout.rollout_stats(policy, wrapped_env, n_episodes=10)
  # Pendulum-v0 always has negative rewards
  assert default_stats['return_max'] < 0
  # ours gives between 1 * traj_len and num_envs * traj_len reward
  # (trajectories are all constant length of 200 in Pendulum)
  steps = wrapped_stats['len_mean']
  assert wrapped_stats['return_min'] == 1 * steps
  assert wrapped_stats['return_max'] == num_envs * steps

  # check that wrapped reward is negative (all pendulum rewards is negative)
  # and other rewards are non-negative
  rand_act, _, _, _ = policy.step(wrapped_env.reset())
  _, rew, _, infos = wrapped_env.step(rand_act)
  assert np.all(rew >= 0)
  assert np.all([info_dict['wrapped_env_rew'] < 0 for info_dict in infos])
