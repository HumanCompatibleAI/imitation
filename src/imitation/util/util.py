"""Miscellaneous utility methods."""

import datetime
import functools
import itertools
import os
import pathlib
import uuid
import warnings
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    TypeVar,
    Union,
    overload,
)

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium.wrappers import TimeLimit
from stable_baselines3.common import monitor, policies
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv

from imitation.data.types import AnyPath


def save_policy(policy: policies.BasePolicy, policy_path: AnyPath) -> None:
    """Save policy to a path.

    Args:
        policy: policy to save.
        policy_path: path to save policy to.
    """
    th.save(policy, parse_path(policy_path))


def oric(x: np.ndarray) -> np.ndarray:
    """Optimal rounding under integer constraints.

    Given a vector of real numbers such that the sum is an integer, returns a vector
    of rounded integers that preserves the sum and which minimizes the Lp-norm of the
    difference between the rounded and original vectors for all p >= 1. Algorithm from
    https://arxiv.org/abs/1501.00014. Runs in O(n log n) time.

    Args:
        x: A 1D vector of real numbers that sum to an integer.

    Returns:
        A 1D vector of rounded integers, preserving the sum.
    """
    rounded = np.floor(x)
    shortfall = x - rounded

    # The total shortfall should be *exactly* an integer, but we
    # round to account for numerical error.
    total_shortfall = np.round(shortfall.sum()).astype(int)
    indices = np.argsort(-shortfall)

    # Apportion the total shortfall to the elements in order of
    # decreasing shortfall.
    rounded[indices[:total_shortfall]] += 1
    return rounded.astype(int)


def make_unique_timestamp() -> str:
    """Timestamp, with random uuid added to avoid collisions."""
    ISO_TIMESTAMP = "%Y%m%d_%H%M%S"
    timestamp = datetime.datetime.now().strftime(ISO_TIMESTAMP)
    random_uuid = uuid.uuid4().hex[:6]
    return f"{timestamp}_{random_uuid}"


def make_vec_env(
    env_name: str,
    *,
    rng: np.random.Generator,
    n_envs: int = 8,
    parallel: bool = False,
    log_dir: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
    post_wrappers: Optional[Sequence[Callable[[gym.Env, int], gym.Env]]] = None,
    env_make_kwargs: Optional[Mapping[str, Any]] = None,
) -> VecEnv:
    """Makes a vectorized environment.

    Args:
        env_name: The Env's string id in Gym.
        rng: The random state to use to seed the environment.
        n_envs: The number of duplicate environments.
        parallel: If True, uses SubprocVecEnv; otherwise, DummyVecEnv.
        log_dir: If specified, saves Monitor output to this directory.
        max_episode_steps: If specified, wraps each env in a TimeLimit wrapper
            with this episode length. If not specified and `max_episode_steps`
            exists for this `env_name` in the Gym registry, uses the registry
            `max_episode_steps` for every TimeLimit wrapper (this automatic
            wrapper is the default behavior when calling `gym.make`). Otherwise
            the environments are passed into the VecEnv unwrapped.
        post_wrappers: If specified, iteratively wraps each environment with each
            of the wrappers specified in the sequence. The argument should be a Callable
            accepting two arguments, the Env to be wrapped and the environment index,
            and returning the wrapped Env.
        env_make_kwargs: The kwargs passed to `spec.make`.

    Returns:
        A VecEnv initialized with `n_envs` environments.
    """
    # Resolve the spec outside of the subprocess first, so that it is available to
    # subprocesses running `make_env` via automatic pickling.
    # Just to ensure packages are imported and spec is properly resolved
    tmp_env = gym.make(env_name)
    tmp_env.close()
    spec = tmp_env.spec
    env_make_kwargs = env_make_kwargs or {}

    def make_env(i: int, this_seed: int) -> gym.Env:
        # Previously, we directly called `gym.make(env_name)`, but running
        # `imitation.scripts.train_adversarial` within `imitation.scripts.parallel`
        # created a weird interaction between Gym and Ray -- `gym.make` would fail
        # inside this function for any of our custom environment unless those
        # environments were also `gym.register()`ed inside `make_env`. Even
        # registering the custom environment in the scope of `make_vec_env` didn't
        # work. For more discussion and hypotheses on this issue see PR #160:
        # https://github.com/HumanCompatibleAI/imitation/pull/160.
        assert env_make_kwargs is not None  # Note: to satisfy mypy
        env = gym.make(spec, max_episode_steps=max_episode_steps, **env_make_kwargs)

        # Seed each environment with a different, non-sequential seed for diversity
        # (even if caller is passing us sequentially-assigned base seeds). int() is
        # necessary to work around gym bug where it chokes on numpy int64s.
        env.reset(seed=int(this_seed))
        # NOTE: we do it here rather than on the final VecEnv, because
        # that would set the same seed for all the environments.

        # Use Monitor to record statistics needed for Baselines algorithms logging
        # Optionally, save to disk
        log_path = None
        if log_dir is not None:
            log_subdir = os.path.join(log_dir, "monitor")
            os.makedirs(log_subdir, exist_ok=True)
            log_path = os.path.join(log_subdir, f"mon{i:03d}")

        env = monitor.Monitor(env, log_path)

        if post_wrappers:
            for wrapper in post_wrappers:
                env = wrapper(env, i)

        return env

    env_seeds = make_seeds(rng, n_envs)
    env_fns: List[Callable[[], gym.Env]] = [
        functools.partial(make_env, i, s) for i, s in enumerate(env_seeds)
    ]
    if parallel:
        # See GH hill-a/stable-baselines issue #217
        return SubprocVecEnv(env_fns, start_method="forkserver")
    else:
        return DummyVecEnv(env_fns)


@overload
def make_seeds(
    rng: np.random.Generator,
) -> int:
    ...


@overload
def make_seeds(rng: np.random.Generator, n: int) -> List[int]:
    ...


def make_seeds(
    rng: np.random.Generator,
    n: Optional[int] = None,
) -> Union[Sequence[int], int]:
    """Generate n random seeds from a random state.

    Args:
        rng: The random state to use to generate seeds.
        n: The number of seeds to generate.

    Returns:
        A list of n random seeds.
    """
    seeds_arr = rng.integers(0, (1 << 31) - 1, (n if n is not None else 1,))
    seeds: List[int] = seeds_arr.tolist()
    if n is None:
        return seeds[0]
    else:
        return seeds


def docstring_parameter(*args, **kwargs):
    """Treats the docstring as a format string, substituting in the arguments."""

    def helper(obj):
        obj.__doc__ = obj.__doc__.format(*args, **kwargs)
        return obj

    return helper


T = TypeVar("T")


def endless_iter(iterable: Iterable[T]) -> Iterator[T]:
    """Generator that endlessly yields elements from `iterable`.

    >>> x = range(2)
    >>> it = endless_iter(x)
    >>> next(it)
    0
    >>> next(it)
    1
    >>> next(it)
    0

    Args:
        iterable: The non-iterator iterable object to endlessly iterate over.

    Returns:
        An iterator that repeats the elements in `iterable` forever.

    Raises:
        ValueError: if iterable is an iterator -- that will be exhausted, so
            cannot be iterated over endlessly.
    """
    if iter(iterable) == iterable:
        raise ValueError("endless_iter needs a non-iterator Iterable.")

    _, iterable = get_first_iter_element(iterable)
    return itertools.chain.from_iterable(itertools.repeat(iterable))


def safe_to_tensor(array: Union[np.ndarray, th.Tensor], **kwargs) -> th.Tensor:
    """Converts a NumPy array to a PyTorch tensor.

    The data is copied in the case where the array is non-writable. Unfortunately if
    you just use `th.as_tensor` for this, an ugly warning is logged and there's
    undefined behavior if you try to write to the tensor.

    Args:
        array: The array to convert to a PyTorch tensor.
        kwargs: Additional keyword arguments to pass to `th.as_tensor`.

    Returns:
        A PyTorch tensor with the same content as `array`.
    """
    if isinstance(array, th.Tensor):
        return array

    if not array.flags.writeable:
        array = array.copy()

    return th.as_tensor(array, **kwargs)


@overload
def safe_to_numpy(obj: Union[np.ndarray, th.Tensor], warn: bool = False) -> np.ndarray:
    ...


@overload
def safe_to_numpy(obj: None, warn: bool = False) -> None:
    ...


def safe_to_numpy(
    obj: Optional[Union[np.ndarray, th.Tensor]],
    warn: bool = False,
) -> Optional[np.ndarray]:
    """Convert torch tensor to numpy.

    If the object is already a numpy array, return it as is.
    If the object is none, returns none.

    Args:
        obj: torch tensor object to convert to numpy array
        warn: if True, warn if the object is not already a numpy array. Useful for
            warning the user of a potential performance hit if a torch tensor is
            not the expected input type.

    Returns:
        Object converted to numpy array
    """
    if obj is None:
        # We ignore the type due to https://github.com/google/pytype/issues/445
        return None  # pytype: disable=bad-return-type
    elif isinstance(obj, np.ndarray):
        return obj
    else:
        if warn:
            warnings.warn(
                "Converted tensor to numpy array, might affect performance. "
                "Make sure this is the intended behavior.",
            )
        return obj.detach().cpu().numpy()


def tensor_iter_norm(
    tensor_iter: Iterable[th.Tensor],
    ord: Union[int, float] = 2,  # noqa: A002
) -> th.Tensor:
    """Compute the norm of a big vector that is produced one tensor chunk at a time.

    Args:
        tensor_iter: an iterable that yields tensors.
        ord: order of the p-norm (can be any int or float except 0 and NaN).

    Returns:
        Norm of the concatenated tensors.

    Raises:
        ValueError: ord is 0 (unsupported).
    """
    if ord == 0:
        raise ValueError("This function cannot compute p-norms for p=0.")
    norms = []
    for tensor in tensor_iter:
        norms.append(th.norm(tensor.flatten(), p=ord))
    norm_tensor = th.as_tensor(norms)
    # Norm of the norms is equal to the norm of the concatenated tensor.
    # th.norm(norm_tensor) = sum(norm**ord for norm in norm_tensor)**(1/ord)
    # = sum(sum(x**ord for x in tensor) for tensor in tensor_iter)**(1/ord)
    # = sum(x**ord for x in tensor for tensor in tensor_iter)**(1/ord)
    # = th.norm(concatenated tensors)
    return th.norm(norm_tensor, p=ord)


def get_first_iter_element(iterable: Iterable[T]) -> Tuple[T, Iterable[T]]:
    """Get first element of an iterable and a new fresh iterable.

    The fresh iterable has the first element added back using ``itertools.chain``.
    If the iterable is not an iterator, this is equivalent to
    ``(next(iter(iterable)), iterable)``.

    Args:
        iterable: The iterable to get the first element of.

    Returns:
        A tuple containing the first element of the iterable, and a fresh iterable
        with all the elements.

    Raises:
        ValueError: `iterable` is empty -- the first call to it returns no elements.
    """
    iterator = iter(iterable)
    try:
        first_element = next(iterator)
    except StopIteration:
        raise ValueError(f"iterable {iterable} had no elements to iterate over.")

    return_iterable: Iterable[T]
    if iterator == iterable:
        # `iterable` was an iterator. Getting `first_element` will have removed it
        # from `iterator`, so we need to add a fresh iterable with `first_element`
        # added back in.
        return_iterable = itertools.chain([first_element], iterator)
    else:
        # `iterable` was not an iterator; we can just return `iterable`.
        # `iter(iterable)` will give a fresh iterator containing the first element.
        # It's preferable to return `iterable` without modification so that users
        # can generate new iterators from it as needed.
        return_iterable = iterable

    return first_element, return_iterable


def parse_path(
    path: AnyPath,
    allow_relative: bool = True,
    base_directory: Optional[pathlib.Path] = None,
) -> pathlib.Path:
    """Parse a path to a `pathlib.Path` object.

    All resulting paths are resolved, absolute paths. If `allow_relative` is True,
    then relative paths are allowed as input, and are resolved relative to the
    current working directory, or relative to `base_directory` if it is
    specified.

    Args:
        path: The path to parse. Can be a string, bytes, or `os.PathLike`.
        allow_relative: If True, then relative paths are allowed as input, and
            are resolved relative to the current working directory. If False,
            an error is raised if the path is not absolute.
        base_directory: If specified, then relative paths are resolved relative
            to this directory, instead of the current working directory.

    Returns:
        A `pathlib.Path` object.

    Raises:
        ValueError: If `allow_relative` is False and the path is not absolute.
        ValueError: If `base_directory` is specified and `allow_relative` is
            False.
    """
    if base_directory is not None and not allow_relative:
        raise ValueError(
            "If `base_directory` is specified, then `allow_relative` must be True.",
        )

    parsed_path: pathlib.Path
    if isinstance(path, pathlib.Path):
        parsed_path = path
    elif isinstance(path, str):
        parsed_path = pathlib.Path(path)
    elif isinstance(path, bytes):
        parsed_path = pathlib.Path(path.decode())
    else:
        parsed_path = pathlib.Path(str(path))

    if parsed_path.is_absolute():
        return parsed_path
    else:
        if allow_relative:
            base_directory = base_directory or pathlib.Path.cwd()
            # relative to current working directory
            return base_directory / parsed_path
        else:
            raise ValueError(f"Path {str(parsed_path)} is not absolute")


def parse_optional_path(
    path: Optional[AnyPath],
    allow_relative: bool = True,
    base_directory: Optional[pathlib.Path] = None,
) -> Optional[pathlib.Path]:
    """Parse an optional path to a `pathlib.Path` object.

    All resulting paths are resolved, absolute paths. If `allow_relative` is True,
    then relative paths are allowed as input, and are resolved relative to the
    current working directory, or relative to `base_directory` if it is
    specified.

    Args:
        path: The path to parse. Can be a string, bytes, or `os.PathLike`.
        allow_relative: If True, then relative paths are allowed as input, and
            are resolved relative to the current working directory. If False,
            an error is raised if the path is not absolute.
        base_directory: If specified, then relative paths are resolved relative
            to this directory, instead of the current working directory.

    Returns:
        A `pathlib.Path` object, or None if `path` is None.
    """
    if path is None:
        return None
    else:
        return parse_path(path, allow_relative, base_directory)


def split_in_half(x: int) -> Tuple[int, int]:
    """Split an integer in half, rounding up.

    This is to ensure that the two halves sum to the original integer.

    Args:
        x: The integer to split.

    Returns:
        A tuple containing the two halves of `x`.
    """
    half = x // 2
    return half, x - half


def clear_screen() -> None:
    """Clears the console screen."""
    if os.name == "nt":  # Windows
        os.system("cls")
    else:
        os.system("clear")
