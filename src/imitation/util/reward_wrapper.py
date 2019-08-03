"""Common wrapper for adding custom reward values to an environment."""
import numpy as np
from stable_baselines.common.vec_env import VecEnvWrapper


class RewardVecEnvWrapper(VecEnvWrapper):
  def __init__(self, venv, reward_fn, *, include_steps=False):
    """A RewardVecEnvWrapper uses a provided reward_fn to replace
    the reward function returned by `step()`.

    Automatically resets the inner VecEnv upon initialization. A tricky part
    about this class is keeping track of the most recent observation from each
    environment.

    Will also include the previous reward given by the inner VecEnv in the
    returned info dict under the `wrapped_env_rew` key.

    Args:
        venv (VecEnv): The VecEnv to wrap.
        reward_fn (Callable): A function that wraps takes in arguments
            (old_obs, act, new_obs) and returns a vector of rewards.
        include_steps (bool): should reward_fn be passed an extra keyword
            argument `steps` indicating the number of actions that have been
            taken in each environment since the last reset? Useful for
            time-dependent reward functions.
    """
    assert not isinstance(venv, RewardVecEnvWrapper)
    super().__init__(venv)
    self.include_steps = include_steps
    self.reward_fn = reward_fn
    self.reset()

  @property
  def envs(self):
    return self.venv.envs

  def reset(self):
    self._old_obs = self.venv.reset()
    self._step_counter = np.zeros((self.num_envs, ), dtype='int')
    return self._old_obs

  def step_async(self, actions):
    self._actions = actions
    return self.venv.step_async(actions)

  def step_wait(self):
    obs, old_rews, dones, infos = self.venv.step_wait()
    # The vecenvs automatically reset the underlying environments once they
    # encounter a `done`, in which case the last observation corresponding to
    # the `done` is dropped. We're going to pull it back out of the info dict!
    obs_fixed = []
    for single_obs, single_done, single_infos in zip(obs, dones, infos):
      if single_done:
        single_obs = single_infos['terminal_observation']
      obs_fixed.append(single_obs)
    if self.include_steps:
      rews = self.reward_fn(self._old_obs,
                            self._actions,
                            obs_fixed,
                            steps=self._step_counter)
    else:
      rews = self.reward_fn(self._old_obs, self._actions, obs_fixed)
    assert len(rews) == len(obs), "must return one rew for each env"
    self._step_counter += 1
    done_mask = np.asarray(dones, dtype='bool').reshape((len(dones), ))
    self._step_counter[done_mask] = 0
    # we can just use obs instead of obs_fixed because on the next iteration
    # after a reset we DO want to access the first observation of the new
    # trajectory, not the last observation of the old trajectory
    self._old_obs = obs
    for info_dict, old_rew in zip(infos, old_rews):
      info_dict['wrapped_env_rew'] = old_rew
    return obs, rews, dones, infos
