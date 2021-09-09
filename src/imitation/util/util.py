import datetime
import functools
import itertools
import os
import uuid
from typing import (
    Any,
    Callable,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    Sequence,
    Type,
    TypeVar,
    Union,
)

import gym
import numpy as np
import stable_baselines3
import torch as th
from gym.wrappers import TimeLimit
from stable_baselines3.common import monitor, preprocessing
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import ActorCriticPolicy, BasePolicy
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv, VecEnv


def make_unique_timestamp() -> str:
    """Timestamp, with random uuid added to avoid collisions."""
    ISO_TIMESTAMP = "%Y%m%d_%H%M%S"
    timestamp = datetime.datetime.now().strftime(ISO_TIMESTAMP)
    random_uuid = uuid.uuid4().hex[:6]
    return f"{timestamp}_{random_uuid}"


def make_vec_env(
    env_name: str,
    n_envs: int = 8,
    seed: int = 0,
    parallel: bool = False,
    log_dir: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
    post_wrappers: Optional[Sequence[Callable[[gym.Env, int], gym.Env]]] = None,
    env_make_kwargs: Optional[Mapping[str, Any]] = None,
) -> VecEnv:
    """Returns a VecEnv initialized with `n_envs` Envs.

    Args:
        env_name: The Env's string id in Gym.
        n_envs: The number of duplicate environments.
        seed: The environment seed.
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
    """
    # Resolve the spec outside of the subprocess first, so that it is available to
    # subprocesses running `make_env` via automatic pickling.
    spec = gym.spec(env_name)
    env_make_kwargs = env_make_kwargs or {}

    def make_env(i, this_seed):
        # Previously, we directly called `gym.make(env_name)`, but running
        # `imitation.scripts.train_adversarial` within `imitation.scripts.parallel`
        # created a weird interaction between Gym and Ray -- `gym.make` would fail
        # inside this function for any of our custom environment unless those
        # environments were also `gym.register()`ed inside `make_env`. Even
        # registering the custom environment in the scope of `make_vec_env` didn't
        # work. For more discussion and hypotheses on this issue see PR #160:
        # https://github.com/HumanCompatibleAI/imitation/pull/160.
        env = spec.make(**env_make_kwargs)

        # Seed each environment with a different, non-sequential seed for diversity
        # (even if caller is passing us sequentially-assigned base seeds). int() is
        # necessary to work around gym bug where it chokes on numpy int64s.
        env.seed(int(this_seed))

        if max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps)
        elif spec.max_episode_steps is not None:
            env = TimeLimit(env, max_episode_steps=spec.max_episode_steps)

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

    rng = np.random.RandomState(seed)
    env_seeds = rng.randint(0, (1 << 31) - 1, (n_envs,))
    env_fns = [functools.partial(make_env, i, s) for i, s in enumerate(env_seeds)]
    if parallel:
        # See GH hill-a/stable-baselines issue #217
        return SubprocVecEnv(env_fns, start_method="forkserver")
    else:
        return DummyVecEnv(env_fns)


# TODO(adam): this is a very simple function; convenient given how Sacred config
# currently works, but we should maybe refactor to avoid needing that.
def init_rl(
    env: Union[gym.Env, VecEnv],
    model_class: Type[BaseAlgorithm] = stable_baselines3.PPO,
    policy_class: Type[BasePolicy] = ActorCriticPolicy,
    **model_kwargs,
):
    """Instantiates a policy for the provided environment.

    Args:
        env: The (vector) environment.
        model_class: A Stable Baselines RL algorithm.
        policy_class: A Stable Baselines compatible policy network class.
        model_kwargs (dict): kwargs passed through to the algorithm.
          Note: anything specified in `policy_kwargs` is passed through by the
          algorithm to the policy network.

    Returns:
      An RL algorithm.
    """
    return model_class(
        policy_class, env, **model_kwargs
    )  # pytype: disable=not-instantiable


def docstring_parameter(*args, **kwargs):
    """Treats the docstring as a format string, substituting in the arguments."""

    def helper(obj):
        obj.__doc__ = obj.__doc__.format(*args, **kwargs)
        return obj

    return helper


T = TypeVar("T")


def endless_iter(iterable: Iterable[T]) -> Iterator[T]:
    """Generator that endlessly yields elements from iterable.

    If any call to `iter(iterable)` has no elements, then this function raises
    ValueError.

    >>> x = range(2)
    >>> it = endless_iter(x)
    >>> next(it)
    0
    >>> next(it)
    1
    >>> next(it)
    0

    """
    try:
        next(iter(iterable))
    except StopIteration:
        err = ValueError(f"iterable {iterable} had no elements to iterate over.")
        raise err

    return itertools.chain.from_iterable(itertools.repeat(iterable))


def torchify_with_space(
    array: np.ndarray,
    space: gym.Space,
    normalize_images: bool = True,
    device: Optional[th.device] = None,
) -> th.Tensor:
    """Converts a `np.ndarray` into `Tensor` corresponding to `space`.

    The shape of the return value may differ from the shape of the input
    value. For example, if `space` is discrete, then the input should be
    an 1D array of scalar values, and the output will be encoded as 2D
    Tensor of one-hot vectors.

    Args:
        array: An array of observations or actions.
        space: The space each value in `array` is sampled from.
        normalize_images: `normalize_images` keyword argument to
          `preprocessing.preprocess_obs`. If True, then image `array`
          is normalized so that each element is between 0 and 1.
        device: Tensor device.
    """
    tensor = th.as_tensor(array, device=device)
    preprocessed = preprocessing.preprocess_obs(
        tensor,
        space,
        # TODO(sam): can I remove "scale" kwarg in DiscrimNet etc.?
        normalize_images=normalize_images,
    )
    return preprocessed


def tensor_iter_norm(
    tensor_iter: Iterable[th.Tensor], ord: Union[int, float] = 2  # noqa: A002
) -> th.Tensor:
    """Compute the norm of a big vector that is produced one tensor chunk at a time.

    Args:
        tensor_iter: an iterable that yields tensors.
        ord: order of the p-norm (can be any int or float except 0 and NaN).

    Returns:
        Norm of the concatenated tensors."""
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
