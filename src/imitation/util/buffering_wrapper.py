import numpy as np
from stable_baselines.common.vec_env import VecEnv, VecEnvWrapper

from imitation.util import rollout


class BufferingWrapper(VecEnvWrapper):
  """Saves transitions of underlying VecEnv.

  Retrieve saved transitions using `pop_transitions()`.
  """

  def __init__(self, venv: VecEnv, error_on_premature_reset: bool = True):
    """
    Args:
      venv: The wrapped VecEnv.
      error_premature_reset: Error if `reset()` is called on this wrapper
        and there are saved samples that haven't yet been accessed.
    """
    super().__init__(venv)
    self.error_on_premature_reset = error_on_premature_reset
    self.obs_list = None
    self.acts_list = None

  def reset(self, **kwargs):
    if self.error_on_premature_reset and len(self.acts_list) != 0:
      raise RuntimeError("TransitionsRecordingWrapper reset() before samples "
                         "were accessed")
    obs = self.venv.reset(**kwargs)
    self.obs_list = [obs]
    self.acts_list = []
    return obs

  def step(self, actions):
    obs, rews, dones, infos = self.venv.step_wait()
    real_obs = np.copy(obs)
    for i, done in enumerate(dones):
      if done:
        real_obs[i] = infos[i]['terminal_observation']
    self.obs_list.append(real_obs)
    self.acts_list.append(actions)
    assert len(self.obs_list) == len(self.acts_list) + 1
    return obs, rews, dones, infos

  def pop_transitions(self) -> rollout.Transitions:
    """Pops recorded transitions, returning them as an instance of Transitions.
    """
    assert len(self.obs_list) == len(self.acts_list) + 1
    obs = np.stack(self.obs_list[:-1])
    acts = np.stack(self.acts_list)
    new_obs = np.stack(self.obs_list[1:])

    self.obs_list = [self.obs_list[-1]]
    self.acts_list = []

    return rollout.Transitions(obs=obs, acts=acts, new_obs=new_obs)

  def step_wait(self):
    return self.venv.step_wait()
