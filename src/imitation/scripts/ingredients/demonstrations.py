"""This ingredient provides (expert) demonstrations to learn from.

The demonstrations are either loaded from disk, from the HuggingFace Dataset Hub, or
sampled from the expert policy provided by the expert ingredient.
"""

import logging
import pathlib
import warnings
from typing import Optional, Sequence, Union

import numpy as np
import sacred

from imitation.data import rollout, serialize, types
from imitation.scripts.ingredients import environment, expert
from imitation.scripts.ingredients import logging as logging_ingredient

demonstrations_ingredient = sacred.Ingredient(
    "demonstrations",
    ingredients=[
        expert.expert_ingredient,
        logging_ingredient.logging_ingredient,
        environment.environment_ingredient,
    ],
)
logger = logging.getLogger(__name__)


@demonstrations_ingredient.config
def config():
    # Either "local" or "{algo}-huggingface" to load them from the HuggingFace Dataset Hub.
    rollout_type = "local"

    # If none, they are sampled from the expert policy.
    rollout_path = None

    # Num demos used or sampled. None loads every demo possible.
    n_expert_demos = None

    locals()  # quieten flake8


@demonstrations_ingredient.named_config
def fast():
    # Note: we can't pick `n_expert_demos=1` here because for envs with short episodes
    #   that does not generate the minimum number of transitions required for one batch.
    n_expert_demos = 10  # noqa: F841


@demonstrations_ingredient.capture
def get_expert_trajectories(
    rollout_type: str,
    rollout_path: str,
) -> Sequence[types.Trajectory]:
    """Loads expert demonstrations.

    Args:
        rollout_type: Can be either `local` to load rollouts from the disk or to
            generate them locally or of the format `{algo}-huggingface` to load
            from the huggingface hub of expert trained using `{algo}`.
        rollout_path: A path containing a pickled sequence of `types.Trajectory`.

    Returns:
        The expert trajectories.

    Raises:
        ValueError: if `rollout_type` is not "local" or of the form {algo}-huggingface.
    """
    if rollout_type.endswith("-huggingface"):
        if rollout_path is not None:
            warnings.warn(
                "Ignoring `rollout_path` since `rollout_type` is set to download the "
                "rollouts from the huggingface-hub. If you want to load the rollouts "
                'from disk, set `rollout_type`="local" and the path in `rollout_path`.',
                RuntimeWarning,
            )
        rollout_path = _download_expert_rollouts(rollout_type)
    elif rollout_type != "local":
        raise ValueError(
            "`rollout_type` can either be `local` or of the form `{algo}-huggingface`.",
        )

    if rollout_path is not None:
        return load_local_expert_trajs(rollout_path)
    else:
        return generate_expert_trajs()


@demonstrations_ingredient.capture
def generate_expert_trajs(
    n_expert_demos: Optional[int],
    _rnd: np.random.Generator,
) -> Optional[Sequence[types.Trajectory]]:
    """Generates expert demonstrations.

    Args:
        n_expert_demos: The number of trajectories to load.
            Dataset is truncated to this length if specified.
        _rnd: Random number generator provided by Sacred.

    Returns:
        The expert trajectories.

    Raises:
        ValueError: If n_expert_demos is None.
    """
    if n_expert_demos is None:
        raise ValueError("n_expert_demos must be specified when rollout_path is None")

    with environment.make_rollout_venv() as rollout_env:
        return rollout.rollout(
            expert.get_expert_policy(rollout_env),
            rollout_env,
            rollout.make_sample_until(min_episodes=n_expert_demos),
            rng=_rnd,
        )


@demonstrations_ingredient.capture
def load_local_expert_trajs(
    rollout_path: Union[str, pathlib.Path],
    n_expert_demos: Optional[int],
) -> Sequence[types.Trajectory]:
    """Loads expert demonstrations from a local path.

    Args:
        rollout_path: A path containing a pickled sequence of `types.Trajectory`.
        n_expert_demos: The number of trajectories to load.
            Dataset is truncated to this length if specified.

    Returns:
        The expert trajectories.

    Raises:
        ValueError: There are fewer trajectories than `n_expert_demos`.
    """
    expert_trajs = serialize.load(rollout_path)
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


@demonstrations_ingredient.capture(prefix="expert")
def _download_expert_rollouts(rollout_type, loader_kwargs):
    assert rollout_type.endswith("-huggingface")
    algo_name = rollout_type.split("-")[0]
    return serialize.load_rollouts_from_huggingface(
        algo_name,
        env_name=loader_kwargs["env_name"],
        organization=loader_kwargs["organization"],
    )
