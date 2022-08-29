"""Common configuration element for scripts learning from demonstrations."""

import logging
from typing import Optional, Sequence

import gym
import sacred
from stable_baselines3.common import vec_env

from imitation.data import rollout, types, wrappers
from imitation.scripts.common import common, expert

demonstrations_ingredient = sacred.Ingredient(
    "demonstrations",
    ingredients=[expert.expert_ingredient, common.common_ingredient],
)
logger = logging.getLogger(__name__)


@demonstrations_ingredient.config
def config():
    # Demonstrations
    rollout_path = None  # path to file containing rollouts
    n_expert_demos = None  # Num demos used. None uses every demo possible
    locals()  # quieten flake8


@demonstrations_ingredient.named_config
def fast():
    n_expert_demos = 1  # noqa: F841


@demonstrations_ingredient.capture
def get_expert_trajectories(
    rollout_path: str,
) -> Sequence[types.Trajectory]:
    if rollout_path is not None:
        return load_expert_trajs()
    else:
        return generate_expert_trajs()


@demonstrations_ingredient.capture
def generate_expert_trajs(
    n_expert_demos: Optional[int],
    common,
) -> Optional[Sequence[types.Trajectory]]:
    """Generates expert demonstrations.

    Args:
        n_expert_demos: The number of trajectories to load.
            Dataset is truncated to this length if specified.
        common: The common config.

    Returns:
        The expert trajectories.

    Raises:
        ValueError: If n_expert_demos is None.
    """
    if n_expert_demos is None:
        raise ValueError("n_expert_demos must be specified when rollout_path is None")

    rollout_env = vec_env.DummyVecEnv(
        [
            lambda: wrappers.RolloutInfoWrapper(gym.make(common["env_name"]))
            for _ in range(4)
        ],
    )
    return rollout.rollout(
        expert.get_expert_policy(),
        rollout_env,
        rollout.make_sample_until(min_timesteps=2000, min_episodes=n_expert_demos),
    )


@demonstrations_ingredient.capture
def load_expert_trajs(
    rollout_path: str,
    n_expert_demos: Optional[int],
) -> Sequence[types.Trajectory]:
    """Loads expert demonstrations.

    Args:
        rollout_path: A path containing a pickled sequence of `types.Trajectory`.
        n_expert_demos: The number of trajectories to load.
            Dataset is truncated to this length if specified.

    Returns:
        The expert trajectories.

    Raises:
        ValueError: There are fewer trajectories than `n_expert_demos`.
    """
    expert_trajs = types.load(rollout_path)
    logger.info(f"Loaded {len(expert_trajs)} expert trajectories from '{rollout_path}'")
    if n_expert_demos is not None:
        if len(expert_trajs) < n_expert_demos:
            raise ValueError(
                f"Want to use n_expert_demos={n_expert_demos} trajectories, but only "
                f"{len(expert_trajs)} are available via {rollout_path}.",
            )
        expert_trajs = expert_trajs[:n_expert_demos]
        logger.info(f"Truncated to {n_expert_demos} expert trajectories")
    return expert_trajs
