"""Custom policy classes and convenience methods."""

import abc

import gym
import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.utils import is_vectorized_observation


class HardCodedPolicy(BasePolicy, abc.ABC):
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
            np_actions.append(self._choose_action(np_obs))
        np_actions = np.stack(np_actions, axis=0)
        th_actions = th.as_tensor(np_actions, device=self.device)
        return th_actions

    @abc.abstractmethod
    def _choose_action(self, obs: np.ndarray) -> np.ndarray:
        """Chooses an action, optionally based on observation obs."""


class RandomPolicy(HardCodedPolicy):
    """Returns random actions."""

    def _choose_action(self, obs: np.ndarray) -> np.ndarray:
        return self.action_space.sample()


class ZeroPolicy(HardCodedPolicy):
    """Returns constant zero action."""

    def _choose_action(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(self.action_space.shape, dtype=self.action_space.dtype)


class FeedForward32Policy(ActorCriticPolicy):
    """A feed forward policy network with two hidden layers of 32 units.

    This matches the IRL policies in the original AIRL paper.

    Note: This differs from stable_baselines MlpPolicy in two ways: by having
    32 rather than 64 units, and by having policy and value networks share weights
    except at the final layer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, net_arch=[32, 32])


def get_action_policy(policy, observation, deterministic=True):
    """Gets an action from a Stable Baselines policy after some processing.

    Specifically, clips actions to the action space associated with `policy` and
    automatically accounts for vectorized environments inputs.

    This code was adapted from Stable Baselines' `BaseAlgorithm.predict()`.

    Args:
        policy (stable_baselines.common.policies.BasePolicy): The policy.
        observation (np.ndarray): The input to the policy network. Can either
            be a single input with shape `policy.observation_space.shape` or a
            vectorized input with shape `(n_batch,) +
            policy.observation_space.shape`.
        deterministic (bool): Whether or not to return deterministic actions
            (usually means argmax over policy's action distribution).

    Returns:
       action (np.ndarray): The action output of the policy network. If
           `observation` is not vectorized (has shape `policy.observation_space.shape`
           instead of shape `(n_batch,) + policy.observation_space.shape`) then
           `action` has shape `policy.action_space.shape`.
           Otherwise, `action` has shape `(n_batch,) + policy.action_space.shape`.
       states(np.ndarray or None): if this policy is recurrent, this will
            return the internal LSTM state at the end of the supplied observation
            sequence. Otherwise, it returns None. As of 2020-07-08, SB3 doesn't
            support RNN policies, so this value is always None.
    """
    # TODO: automate type conversions. There is probably something in SB3 to do
    # this.
    is_vec_obs = is_vectorized_observation(observation, policy.observation_space)

    observation = observation.reshape((-1,) + policy.observation_space.shape)
    with th.no_grad():
        # returns (actions, values, log_prob)
        actions, states = policy.predict(observation, deterministic=deterministic)

    clipped_actions = actions
    if isinstance(policy.action_space, gym.spaces.Box):
        clipped_actions = np.clip(
            actions, policy.action_space.low, policy.action_space.high
        )

    if not is_vec_obs:
        clipped_actions = clipped_actions[0]

    return clipped_actions, states
