"""Custom policy classes and convenience methods."""

import abc
from typing import Type

import gym
import numpy as np
import torch as th
from stable_baselines3.common import policies, torch_layers
from stable_baselines3.sac import policies as sac_policies
from torch import nn

from imitation.util import networks


class NonTrainablePolicy(policies.BasePolicy, abc.ABC):
    """Abstract class for non-trainable (e.g. hard-coded or interactive) policies."""

    def __init__(self, observation_space: gym.Space, action_space: gym.Space):
        """Builds NonTrainablePolicy with specified observation and action space."""
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
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
        raise NotImplementedError  # pragma: no cover


class RandomPolicy(NonTrainablePolicy):
    """Returns random actions."""

    def _choose_action(self, obs: np.ndarray) -> np.ndarray:
        return self.action_space.sample()


class ZeroPolicy(NonTrainablePolicy):
    """Returns constant zero action."""

    def _choose_action(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(self.action_space.shape, dtype=self.action_space.dtype)


class FeedForward32Policy(policies.ActorCriticPolicy):
    """A feed forward policy network with two hidden layers of 32 units.

    This matches the IRL policies in the original AIRL paper.

    Note: This differs from stable_baselines3 ActorCriticPolicy in two ways: by
    having 32 rather than 64 units, and by having policy and value networks
    share weights except at the final layer, where there are different linear heads.
    """

    def __init__(self, *args, **kwargs):
        """Builds FeedForward32Policy; arguments passed to `ActorCriticPolicy`."""
        super().__init__(*args, **kwargs, net_arch=[32, 32])


class SAC1024Policy(sac_policies.SACPolicy):
    """Actor and value networks with two hidden layers of 1024 units respectively.

    This matches the implementation of SAC policies in the PEBBLE paper. See:
    https://arxiv.org/pdf/2106.05091.pdf
    https://github.com/denisyarats/pytorch_sac/blob/master/config/agent/sac.yaml

    Note: This differs from stable_baselines3 SACPolicy by having 1024 hidden units
    in each layer instead of the default value of 256.
    """

    def __init__(self, *args, **kwargs):
        """Builds SAC1024Policy; arguments passed to `SACPolicy`."""
        super().__init__(*args, **kwargs, net_arch=[1024, 1024])


class NormalizeFeaturesExtractor(torch_layers.FlattenExtractor):
    """Feature extractor that flattens then normalizes input."""

    def __init__(
        self,
        observation_space: gym.Space,
        normalize_class: Type[nn.Module] = networks.RunningNorm,
    ):
        """Builds NormalizeFeaturesExtractor.

        Args:
            observation_space: The space observations lie in.
            normalize_class: The class to use to normalize observations (after being
                flattened). This can be any Module that preserves the shape;
                e.g. `nn.BatchNorm*` or `nn.LayerNorm`.
        """
        super().__init__(observation_space)
        # Below we have to ignore the type error when initializing the class because
        # there is no simple way of specifying a protocol that admits one positional
        # argument for the number of features while being compatible with nn.Module.
        # (it would require defining a base class and forcing all the subclasses
        # to inherit from it).
        self.normalize = normalize_class(self.features_dim)  # type: ignore[call-arg]

    def forward(self, observations: th.Tensor) -> th.Tensor:
        flattened = super().forward(observations)
        return self.normalize(flattened)
