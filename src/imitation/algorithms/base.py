import abc
from typing import Iterable, Optional

from imitation.data import types
from imitation.util import logger as imit_logger


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
        custom_logger: Optional[imit_logger.HierarchicalLogger],
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
                https://imitation.readthedocs.io/en/latest/guide/variable_horizon.html
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
                "https://imitation.readthedocs.io/en/latest/guide/variable_horizon.html"
                " for more information.",
            )
        self._horizon = None

    @property
    def logger(self):
        return self._logger

    @logger.setter
    def logger(self, value: imit_logger.HierarchicalLogger) -> None:
        self._logger = value

    def _check_fixed_horizon(self, trajs: Iterable[types.Trajectory]) -> None:
        """Check that `trajs` has fixed episode length and equal to prior calls.

        If algorithm is safe to use with variable horizon episodes (e.g. behavioral
        cloning), then just don't call this method.

        Args:
            trajs: An iterable sequence of trajectories.

        Raises:
            ValueError if the length of trajectories in trajs are different from one
            another, or from trajectory lengths in previous calls to this method.
        """
        if self.allow_variable_horizon:  # skip check -- YOLO
            return

        # horizons = all horizons seen so far (including trajs)
        horizons = set(len(traj) for traj in trajs if traj.terminal)
        if self._horizon is not None:
            horizons.add(self._horizon)

        if len(horizons) > 1:
            raise ValueError(
                f"Episodes of different length detected: {horizons}. "
                "Variable horizon environments are discouraged -- "
                "termination conditions leak information about reward. See"
                "https://imitation.readthedocs.io/en/latest/guide/variable_horizon.html"
                " for more information. If you are SURE you want to run imitation on a "
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
