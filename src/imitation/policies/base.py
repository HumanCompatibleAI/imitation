"""Custom policy classes and convenience methods."""

import abc

import gym
import numpy as np
import torch as th
from stable_baselines3.common import policies


class HardCodedPolicy(policies.BasePolicy, abc.ABC):
    """Abstract class for hard-coded (non-trainable) policies."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            device=th.device("cpu"),
        )

    def _predict(self, obs: th.Tensor, deterministic: bool = False):
        np_actions = []
        np_obs = obs.detach().cpu().numpy()
        for np_ob in np_obs:
            assert self.observation_space.contains(np_ob)
            np_actions.append(self._choose_action(np_ob))
        np_actions = np.stack(np_actions, axis=0)
        th_actions = th.as_tensor(np_actions, device=self.device)
        return th_actions

    @abc.abstractmethod
    def _choose_action(self, obs: np.ndarray) -> np.ndarray:
        """Chooses an action, optionally based on observation obs."""

    def forward(self, *args):
        # technically BasePolicy is a Torch module, so this needs a forward()
        # method
        raise NotImplementedError()


class RandomPolicy(HardCodedPolicy):
    """Returns random actions."""

    def _choose_action(self, obs: np.ndarray) -> np.ndarray:
        return self.action_space.sample()


class ZeroPolicy(HardCodedPolicy):
    """Returns constant zero action."""

    def _choose_action(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(self.action_space.shape, dtype=self.action_space.dtype)


class FeedForward32Policy(policies.ActorCriticPolicy):
    """A feed forward policy network with two hidden layers of 32 units.

    This matches the IRL policies in the original AIRL paper.

    Note: This differs from stable_baselines3 ActorCriticPolicy in two ways: by
    having 32 rather than 64 units, and by having policy and value networks
    share weights except at the final layer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, net_arch=[32, 32])
