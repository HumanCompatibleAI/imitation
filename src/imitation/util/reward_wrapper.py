# TODO(shwang, qxcv): Keep Sam's version when merging #54.
class RewardVecEnvWrapper(VecEnvWrapper):

  def __init__(self, venv, reward_fn):
    """A RewardVecEnvWrapper uses a provided reward_fn to replace
    the reward function returned by `step()`.

    Automatically resets the inner VecEnv upon initialization.
    A tricky part about this class keeping track of the most recent
    observation from each environment.

    Args:
        venv (VecEnv): The VecEnv to wrap.
        reward_fn (Callable): A function that wraps takes in arguments
            (old_obs, act, new_obs) and returns a vector of rewards.
    """
    assert not isinstance(venv, RewardVecEnvWrapper)
    super().__init__(venv)
    self.reward_fn = reward_fn
    self.reset()

  @property
  def envs(self):
    return self.venv.envs

  def reset(self):
    self._old_obs = self.venv.reset()
    return self._old_obs

  def step_async(self, actions):
    self._actions = actions
    return self.venv.step_async(actions)

  def step_wait(self):
    obs, rew, done, info = self.venv.step_wait()
    rew = self.reward_fn(self._old_obs, self._actions, obs)
    # XXX: We never get to see episode end. (See Issue #1).
    # Therefore, the final obs of every episode is incorrect.
    self._old_obs = obs
    return obs, rew, done, info
