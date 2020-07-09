"""Custom policy classes and convenience methods."""

import abc

import gym
import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines.common.policies import BasePolicy


class HardCodedPolicy(BasePolicy, abc.ABC):
    """Abstract class for hard-coded (non-trainable) policies."""

    def __init__(self, ob_space: gym.Space, ac_space: gym.Space):
        self.ob_space = ob_space
        self.ac_space = ac_space

    def predict(self, obs, state=None, mask=None, deterministic=False):
        actions = []
        for ob in obs:
            assert self.ob_space.contains(ob)
            actions.append(self._choose_action(obs))
        return np.array(actions), None

    @abc.abstractmethod
    def _choose_action(self, obs: np.ndarray) -> np.ndarray:
        """Chooses an action, optionally based on observation obs."""


class RandomPolicy(HardCodedPolicy):
    """Returns random actions."""

    def __init__(self, ob_space: gym.Space, ac_space: gym.Space):
        super().__init__(ob_space, ac_space)

    def _choose_action(self, obs: np.ndarray) -> np.ndarray:
        return self.ac_space.sample()


class ZeroPolicy(HardCodedPolicy):
    """Returns constant zero action."""

    def __init__(self, ob_space: gym.Space, ac_space: gym.Space):
        super().__init__(ob_space, ac_space)

    def _choose_action(self, obs: np.ndarray) -> np.ndarray:
        return np.zeros(self.ac_space.shape, dtype=self.ac_space.dtype)


class FeedForward32Policy(ActorCriticPolicy):
    """A feed forward policy network with two hidden layers of 32 units.

    This matches the IRL policies in the original AIRL paper.

    Note: This differs from stable_baselines~=2.10.0 MlpPolicy in two ways: by having
    32 rather than 64 units, and by having policy and value networks share weights
    except at the final layer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, net_arch=[32, 32])


# TODO remove the use of this helper function to complete Stable Baselines 3 port
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
