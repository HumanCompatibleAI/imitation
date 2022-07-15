"""Load serialized reward functions of different types."""

from typing import Callable, Iterable, Type

import numpy as np
import torch as th
from stable_baselines3.common.vec_env import VecEnv

from imitation.rewards import reward_function
from imitation.rewards.reward_nets import (
    NormalizedRewardNet,
    RewardNet,
    RewardNetWrapper,
    ShapedRewardNet,
)
from imitation.util import registry, util

# TODO(sam): I suspect this whole file can be replaced with th.load calls. Try
# that refactoring once I have things running.

RewardFnLoaderFn = Callable[[str, VecEnv], reward_function.RewardFn]

reward_registry: registry.Registry[RewardFnLoaderFn] = registry.Registry()


class ValidateRewardFn(reward_function.RewardFn):
    """Wrap reward function to add sanity check.

    Checks that the reward vector is equal to the batch size of the input.
    """

    def __init__(
        self,
        reward_fn: reward_function.RewardFn,
    ):
        """Builds the reward validator.

        Args:
            reward_fn: base reward function
        """
        super().__init__()
        self.reward_fn = reward_fn

    def __call__(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:
        rew = self.reward_fn(state, action, next_state, done)
        assert rew.shape == (len(state),)
        return rew


def _strip_wrappers(
    reward_net: RewardNet,
    wrapper_types: Iterable[Type[RewardNetWrapper]],
) -> RewardNet:
    """Attempts to remove provided wrappers.

    Strips wrappers of type `wrapper_type` from `reward_net` in order until either the
    wrapper type to remove does not match the type of net or there are no more wrappers
    to remove.

    Args:
        reward_net: an instance of a reward network that may be wrapped
        wrapper_types: an iterable of wrapper types in the order they should be removed

    Returns:
        The reward network with the listed wrappers removed
    """
    for wrapper_type in wrapper_types:
        assert issubclass(
            wrapper_type,
            RewardNetWrapper,
        ), f"trying to remove non-wrapper type {wrapper_type}"

        if isinstance(reward_net, wrapper_type):
            reward_net = reward_net.base
        else:
            break

    return reward_net


def _make_functional(
    net: RewardNet,
    attr: str = "predict",
    kwargs=dict(),
) -> reward_function.RewardFn:
    return lambda *args: getattr(net, attr)(*args, **kwargs)


def _validate_type(net: RewardNet, type_: Type[RewardNet]):
    if not isinstance(net, type_):
        raise TypeError(f"expected {type_} but found {type(net)}")
    return net


def load_zero(path: str, venv: VecEnv) -> reward_function.RewardFn:
    del path, venv

    def f(
        obs: np.ndarray,
        act: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        del act, next_obs, dones  # Unused.
        return np.zeros(obs.shape[0])

    return f


# TODO(adam): I think we can get rid of this and have just one RewardNet.

reward_registry.register(
    key="RewardNet_shaped",
    value=lambda path, _: ValidateRewardFn(
        _make_functional(_validate_type(th.load(str(path)), ShapedRewardNet)),
    ),
)

reward_registry.register(
    key="RewardNet_unshaped",
    value=lambda path, _: ValidateRewardFn(
        _make_functional(_strip_wrappers(th.load(str(path)), (ShapedRewardNet,))),
    ),
)

reward_registry.register(
    key="RewardNet_normalized",
    value=lambda path, _: ValidateRewardFn(
        _make_functional(
            _validate_type(th.load(str(path)), NormalizedRewardNet),
            attr="predict_processed",
            kwargs={"update_stats": False},
        ),
    ),
)

reward_registry.register(
    key="RewardNet_unnormalized",
    value=lambda path, _: ValidateRewardFn(
        _make_functional(_strip_wrappers(th.load(str(path)), (NormalizedRewardNet,))),
    ),
)

reward_registry.register(key="zero", value=load_zero)


@util.docstring_parameter(reward_types=", ".join(reward_registry.keys()))
def load_reward(
    reward_type: str,
    reward_path: str,
    venv: VecEnv,
) -> reward_function.RewardFn:
    """Load serialized reward.

    Args:
        reward_type: A key in `reward_registry`. Valid types
            include {reward_types}.
        reward_path: A path specifying the reward.
        venv: An environment that the policy is to be used with.

    Returns:
        The deserialized reward.
    """
    reward_loader = reward_registry.get(reward_type)
    return reward_loader(reward_path, venv)
