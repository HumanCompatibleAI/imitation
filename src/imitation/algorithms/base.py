"""Module of base classes and helper methods for imitation learning algorithms."""

import abc
from typing import (
    Any,
    Generic,
    Iterable,
    Iterator,
    Mapping,
    Optional,
    TypeVar,
    Union,
    cast,
)

import torch as th
import torch.utils.data as th_data
from stable_baselines3.common import policies

from imitation.data import rollout, types
from imitation.util import logger as imit_logger
from imitation.util import util


class BaseImitationAlgorithm(abc.ABC):
    """Base class for all imitation learning algorithms."""

    _logger: imit_logger.HierarchicalLogger
    """Object to log statistics and natural language messages to."""

    allow_variable_horizon: bool
    """If True, allow variable horizon trajectories; otherwise error if detected."""

    _horizon: Optional[int]
    """Horizon of trajectories seen so far (None if no trajectories seen)."""

    def __init__(
        self,
        *,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        allow_variable_horizon: bool = False,
    ):
        """Creates an imitation learning algorithm.

        Args:
            custom_logger: Where to log to; if None (default), creates a new logger.
            allow_variable_horizon: If False (default), algorithm will raise an
                exception if it detects trajectories of different length during
                training. If True, overrides this safety check. WARNING: variable
                horizon episodes leak information about the reward via termination
                condition, and can seriously confound evaluation. Read
                https://imitation.readthedocs.io/en/latest/getting-started/variable-horizon.html
                before overriding this.
        """
        self._logger = custom_logger or imit_logger.configure()
        self.allow_variable_horizon = allow_variable_horizon
        if allow_variable_horizon:
            self.logger.warn(
                "Running with `allow_variable_horizon` set to True. "
                "Some algorithms are biased towards shorter or longer "
                "episodes, which may significantly confound results. "
                "Additionally, even unbiased algorithms can exploit "
                "the information leak from the termination condition, "
                "producing spuriously high performance. See "
                "https://imitation.readthedocs.io/en/latest/getting-started/"
                "variable-horizon.html for more information.",
            )
        self._horizon = None

    @property
    def logger(self) -> imit_logger.HierarchicalLogger:
        return self._logger

    @logger.setter
    def logger(self, value: imit_logger.HierarchicalLogger) -> None:
        self._logger = value

    def _check_fixed_horizon(self, horizons: Iterable[int]) -> None:
        """Checks that episode lengths in `horizons` are fixed and equal to prior calls.

        If algorithm is safe to use with variable horizon episodes (e.g. behavioral
        cloning), then just don't call this method.

        Args:
            horizons: An iterable sequence of episode lengths.

        Raises:
            ValueError: The length of trajectories in trajs differs from one
                another, or from trajectory lengths in previous calls to this method.
        """
        if self.allow_variable_horizon:  # skip check -- YOLO
            return

        # horizons = all horizons seen so far (including trajs)
        horizons = set(horizons)
        if self._horizon is not None:
            horizons.add(self._horizon)

        if len(horizons) > 1:
            raise ValueError(
                f"Episodes of different length detected: {horizons}. "
                "Variable horizon environments are discouraged -- "
                "termination conditions leak information about reward. See "
                "https://imitation.readthedocs.io/en/latest/getting-started/"
                "variable-horizon.html for more information. "
                "If you are SURE you want to run imitation on a "
                "variable horizon task, then please pass in the flag: "
                "`allow_variable_horizon=True`.",
            )
        elif len(horizons) == 1:
            self._horizon = horizons.pop()

    def __getstate__(self):
        state = self.__dict__.copy()
        # logger can't be pickled as it depends on open files
        del state["_logger"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        # callee should modify self.logger directly if they want to override this
        self.logger = state.get("_logger") or imit_logger.configure()


TransitionKind = TypeVar("TransitionKind", bound=types.TransitionsMinimal)
AnyTransitions = Union[
    Iterable[types.Trajectory],
    Iterable[types.TransitionMapping],
    types.TransitionsMinimal,
]


class DemonstrationAlgorithm(BaseImitationAlgorithm, Generic[TransitionKind]):
    """An algorithm that learns from demonstration: BC, IRL, etc."""

    def __init__(
        self,
        *,
        demonstrations: Optional[AnyTransitions],
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        allow_variable_horizon: bool = False,
    ):
        """Creates an algorithm that learns from demonstrations.

        Args:
            demonstrations: Demonstrations from an expert (optional). Transitions
                expressed directly as a `types.TransitionsMinimal` object, a sequence
                of trajectories, or an iterable of transition batches (mappings from
                keywords to arrays containing observations, etc).
            custom_logger: Where to log to; if None (default), creates a new logger.
            allow_variable_horizon: If False (default), algorithm will raise an
                exception if it detects trajectories of different length during
                training. If True, overrides this safety check. WARNING: variable
                horizon episodes leak information about the reward via termination
                condition, and can seriously confound evaluation. Read
                https://imitation.readthedocs.io/en/latest/getting-started/variable-horizon.html
                before overriding this.
        """
        super().__init__(
            custom_logger=custom_logger,
            allow_variable_horizon=allow_variable_horizon,
        )

        if demonstrations is not None:
            self.set_demonstrations(demonstrations)

    @abc.abstractmethod
    def set_demonstrations(self, demonstrations: AnyTransitions) -> None:
        """Sets the demonstration data.

        Changing the demonstration data on-demand can be useful for
        interactive algorithms like DAgger.

        Args:
             demonstrations: Either a Torch `DataLoader`, any other iterator that
                yields dictionaries containing "obs" and "acts" Tensors or NumPy arrays,
                `TransitionKind` instance, or a Sequence of Trajectory objects.
        """

    @property
    @abc.abstractmethod
    def policy(self) -> policies.BasePolicy:
        """Returns a policy imitating the demonstration data."""

    def save_policy(self, policy_path: types.AnyPath) -> None:
        """Save policy to a path.

        Args:
            policy_path: path to save policy to.
        """
        th.save(self.policy, util.parse_path(policy_path))


class _WrappedDataLoader:
    """Wraps a data loader (batch iterable) and checks for specified batch size."""

    def __init__(
        self,
        data_loader: Iterable[types.TransitionMapping],
        expected_batch_size: int,
    ):
        """Builds _WrappedDataLoader.

        Args:
            data_loader: The data loader (batch iterable) to wrap.
            expected_batch_size: The batch size to check for.
        """
        self.data_loader = data_loader
        self.expected_batch_size = expected_batch_size

    def __iter__(self) -> Iterator[types.TransitionMapping]:
        """Yields data from `self.data_loader`, checking `self.expected_batch_size`.

        Yields:
            Identity -- yields same batches as from `self.data_loader`.

        Raises:
            ValueError: `self.data_loader` returns a batch of size not equal to
                `self.expected_batch_size`.
        """
        for batch in self.data_loader:
            if len(batch["obs"]) != self.expected_batch_size:
                raise ValueError(
                    f"Expected batch size {self.expected_batch_size} "
                    f"!= {len(batch['obs'])} = len(batch['obs'])",
                )
            if len(batch["acts"]) != self.expected_batch_size:
                raise ValueError(
                    f"Expected batch size {self.expected_batch_size} "
                    f"!= {len(batch['acts'])} = len(batch['acts'])",
                )
            yield batch


def make_data_loader(
    transitions: AnyTransitions,
    batch_size: int,
    data_loader_kwargs: Optional[Mapping[str, Any]] = None,
) -> Iterable[types.TransitionMapping]:
    """Converts demonstration data to Torch data loader.

    Args:
        transitions: Transitions expressed directly as a `types.TransitionsMinimal`
            object, a sequence of trajectories, or an iterable of transition
            batches (mappings from keywords to arrays containing observations, etc).
        batch_size: The size of the batch to create. Does not change the batch size
            if `transitions` is already an iterable of transition batches.
        data_loader_kwargs: Arguments to pass to `th_data.DataLoader`.

    Returns:
        An iterable of transition batches.

    Raises:
        ValueError: if `transitions` is an iterable over transition batches with batch
            size not equal to `batch_size`; or if `transitions` is transitions or a
            sequence of trajectories with total timesteps less than `batch_size`.
        TypeError: if `transitions` is an unsupported type.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size={batch_size} must be positive.")

    if isinstance(transitions, Iterable):
        # Inferring the correct type here is difficult with generics.
        (
            first_item,
            transitions,
        ) = util.get_first_iter_element(  # type: ignore[assignment]
            transitions,
        )
        if isinstance(first_item, types.Trajectory):
            transitions = cast(Iterable[types.Trajectory], transitions)
            transitions = rollout.flatten_trajectories(list(transitions))

    if isinstance(transitions, types.TransitionsMinimal):
        if len(transitions) < batch_size:
            raise ValueError(
                f"Number of transitions in `demonstrations` {len(transitions)} "
                f"is smaller than batch size {batch_size}.",
            )

        kwargs: Mapping[str, Any] = {
            "shuffle": True,
            "drop_last": True,
            **(data_loader_kwargs or {}),
        }
        return th_data.DataLoader(
            transitions,
            batch_size=batch_size,
            collate_fn=types.transitions_collate_fn,
            **kwargs,
        )
    elif isinstance(transitions, Iterable):
        # Safe to ignore this error since we've already converted Iterable[Trajectory]
        # `transitions` into Iterable[TransitionMapping]
        return _WrappedDataLoader(transitions, batch_size)  # type: ignore[arg-type]
    else:
        raise TypeError(f"`demonstrations` unexpected type {type(transitions)}")
