"""Custom policy classes and convenience methods."""

import abc

import gym
import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy


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


# TODO(scottemmons) remove the use of this helper function to complete Stable
# Baselines 3 port
def get_action_policy(policy, *args, **kwargs):
    """Gets an action from a Stable Baselines policy.

    In a previous version of Stable Baselines, this helper function was needed to do
    processing of the policy's output. However, Stable Baselines 3 handles the
    processing automatically, so this function is now simply an alias for
    policy.predict().

    Args:
        policy (stable_baselines.common.policies.BasePolicy): The policy.
        *args: Positional arguments to pass to policy.predict()
        **kwargs: Keywords arguments to pass to policy.predict()

    Returns:
        (Tuple[np.ndarray, Optional[np.ndarray]]) the model's action and the next state
            (used in recurrent policies)
    """
    return policy.predict(*args, **kwargs)
