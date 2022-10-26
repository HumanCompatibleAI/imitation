"""Load serialized reward functions of different types."""

from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Type, Union, cast

import numpy as np
import torch as th
from stable_baselines3.common.vec_env import VecEnv

from imitation.rewards import reward_function, reward_nets
from imitation.util import registry, util

# TODO(sam): I suspect this whole file can be replaced with th.load calls. Try
# that refactoring once I have things running.

RewardFnLoaderFn = Callable[[str, VecEnv], reward_function.RewardFn]

reward_registry: registry.Registry[RewardFnLoaderFn] = registry.Registry()


class ValidateRewardFn(reward_function.RewardFn):
    """Wrap reward function to add sanity check.

    Checks that the length of the reward vector is equal to the batch size of the input.
    """

    def __init__(
        self,
        reward_fn: reward_function.RewardFn,
    ) -> None:
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
    reward_net: reward_nets.RewardNet,
    wrapper_types: Iterable[Type[reward_nets.RewardNetWrapper]],
) -> reward_nets.RewardNet:
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
            reward_nets.RewardNetWrapper,
        ), f"trying to remove non-wrapper type {wrapper_type}"

        if isinstance(reward_net, wrapper_type):
            reward_net = reward_net.base
        else:
            break

    return reward_net


def _make_functional(
    net: reward_nets.RewardNet,
    attr: str = "predict",
    default_kwargs: Optional[Dict[str, Any]] = None,
    **kwargs: Any,
) -> reward_function.RewardFn:
    if default_kwargs is None:
        default_kwargs = {}
    default_kwargs.update(kwargs)
    return lambda *args: getattr(net, attr)(*args, **default_kwargs)


WrapperPrefix = Sequence[Type[reward_nets.RewardNet]]


def _prefix_matches(wrappers: Sequence[Type[Any]], prefix: Sequence[Type[Any]]) -> bool:
    """Return true if `prefix` is a prefix of `wrappers`."""
    # Base cases
    if len(prefix) == 0:
        # If we run out of prefix before running out of wrappers
        return True
    elif len(wrappers) == 0:
        # If we run out of wrappers before we run out of prefix
        return False

    prefix_head, *prefix_tail = prefix
    wrappers_head, *wrappers_tail = wrappers

    if not issubclass(wrappers_head, prefix_head):
        return False

    return _prefix_matches(wrappers_tail, prefix_tail)


def _validate_wrapper_structure(
    reward_net: Union[reward_nets.RewardNet, reward_nets.RewardNetWrapper],
    prefixes: Iterable[WrapperPrefix],
) -> reward_nets.RewardNet:
    """Reward net if it has a valid structure.

    A wrapper prefix specifies, from outermost to innermost, which wrappers must
    be present. If any of the wrapper prefixes match then the RewardNet is considered
    valid.

    Args:
        reward_net: net to test
        prefixes: A list of acceptable wrapper prefixes.

    Returns:
        the reward_net if it is valid

    Raises:
        TypeError: if the wrapper structure is not valid with a useful message.

    >>> class RewardNetA(RewardNet):
    ...     def forward(*args):
    ...         pass
    >>> class WrapperB(RewardNetWrapper):
    ...     def forward(*args):
    ...         pass
    >>> reward_net = RewardNetA(None, None)
    >>> reward_net = WrapperB(reward_net)
    >>> assert isinstance(reward_net.base, RewardNet)
    >>> reward_net == _validate_wrapper_structure(reward_net, [[WrapperB, RewardNetA]]))
    True
    """
    wrapper = reward_net
    wrappers = []
    while hasattr(wrapper, "base"):
        wrappers.append(wrapper.__class__)
        wrapper = cast(reward_nets.RewardNet, wrapper.base)
    wrappers.append(wrapper.__class__)  # append the final reward net

    if any(_prefix_matches(wrappers, prefix) for prefix in prefixes):
        return reward_net

    # Otherwise provide a useful error
    formatted_prefixes = [
        "[" + ",".join(t.__name__ for t in prefix) + "]" for prefix in prefixes
    ]

    formatted_wrapper_structure = "[" + ",".join(t.__name__ for t in wrappers) + "]"

    raise TypeError(
        "Wrapper structure should"
        + " match "
        + " or ".join(formatted_prefixes)
        + " but found "
        + formatted_wrapper_structure,
    )


def load_zero(path: str, venv: VecEnv) -> reward_function.RewardFn:
    del path, venv

    def f(
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:
        del action, next_state, done  # Unused.
        return np.zeros(state.shape[0])

    return f


# TODO(adam): I think we can get rid of this and have just one RewardNet.

reward_registry.register(
    key="RewardNet_shaped",
    value=lambda path, _, **kwargs: ValidateRewardFn(
        _make_functional(
            _validate_wrapper_structure(
                th.load(str(path)),
                {(reward_nets.ShapedRewardNet,)},
            ),
        ),
    ),
)

reward_registry.register(
    key="RewardNet_unshaped",
    value=lambda path, _, **kwargs: ValidateRewardFn(
        _make_functional(
            _strip_wrappers(th.load(str(path)), (reward_nets.ShapedRewardNet,)),
        ),
    ),
)

reward_registry.register(
    key="RewardNet_normalized",
    value=lambda path, _, **kwargs: ValidateRewardFn(
        _make_functional(
            _validate_wrapper_structure(
                th.load(str(path)),
                {(reward_nets.NormalizedRewardNet,)},
            ),
            attr="predict_processed",
            default_kwargs={"update_stats": False},
            **kwargs,
        ),
    ),
)

reward_registry.register(
    key="RewardNet_unnormalized",
    value=lambda path, _, **kwargs: ValidateRewardFn(
        _make_functional(
            _strip_wrappers(th.load(str(path)), (reward_nets.NormalizedRewardNet,)),
        ),
    ),
)

reward_registry.register(
    key="RewardNet_std_added",
    value=lambda path, _, **kwargs: ValidateRewardFn(
        _make_functional(
            _strip_wrappers(
                _validate_wrapper_structure(
                    th.load(str(path)),
                    {
                        (reward_nets.AddSTDRewardWrapper,),
                        (
                            reward_nets.NormalizedRewardNet,
                            reward_nets.AddSTDRewardWrapper,
                        ),
                    },
                ),
                (reward_nets.NormalizedRewardNet,),
            ),
            attr="predict_processed",
            default_kwargs={},
            **kwargs,
        ),
    ),
)

reward_registry.register(key="zero", value=load_zero)


@util.docstring_parameter(reward_types=", ".join(reward_registry.keys()))
def load_reward(
    reward_type: str,
    reward_path: str,
    venv: VecEnv,
    **kwargs: Any,
) -> reward_function.RewardFn:
    """Load serialized reward.

    Args:
        reward_type: A key in `reward_registry`. Valid types
            include {reward_types}.
        reward_path: A path specifying the reward.
        venv: An environment that the policy is to be used with.
        **kwargs: kwargs to pass to reward fn

    Returns:
        The deserialized reward.
    """
    reward_loader = reward_registry.get(reward_type)
    return reward_loader(reward_path, venv, **kwargs)
