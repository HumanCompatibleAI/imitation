"""Ingredient for scripts learning from demonstrations."""

import logging
from typing import Any, Dict, Optional, Sequence

import datasets
import huggingface_sb3 as hfsb3
import numpy as np
import sacred

from imitation.data import huggingface_utils, rollout, serialize, types
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
    # Either "local" or "huggingface" or "generated".
    source = "generated"

    # local path or huggingface repo id to load rollouts from.
    path = None

    # passed to `datasets.load_dataset` if `source` is "huggingface"
    loader_kwargs: Dict[str, Any] = dict(
        split="train",
    )

    # Used to deduce HuggingFace repo id if `path` is None
    organization = "HumanCompatibleAI"

    # Used to deduce HuggingFace repo id if `path` is None
    algo_name = "ppo"

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
    source: str,
    path: str,
) -> Sequence[types.Trajectory]:
    """Loads expert demonstrations.

    Args:
        source: Can be either `local` to load rollouts from the disk,
            `huggingface` to load from the HuggingFace hub or
            `generated` to generate the expert trajectories.
        path: A path containing a pickled sequence of `sources.Trajectory`.

    Returns:
        The expert trajectories.

    Raises:
        ValueError: if `source` is not in ["local", "huggingface", "generated"].
    """
    if source == "local":
        if path is None:
            raise ValueError(
                "When source is 'local', path must be set.",
            )
        return _constrain_number_of_demos(serialize.load(path))

    if source == "huggingface":
        return _constrain_number_of_demos(_download_expert_rollouts())

    if source == "generated":
        if path is not None:
            logger.warning("Ignoring path when source is 'generated'")
        return _generate_expert_trajs()

    raise ValueError(
        "`source` can either be `local` or `huggingface` or `generated`.",
    )


@demonstrations_ingredient.capture
def _constrain_number_of_demos(
    demos: Sequence[types.Trajectory],
    n_expert_demos: Optional[int],
) -> Sequence[types.Trajectory]:
    """Constrains the number of demonstrations to n_expert_demos if it is not None."""
    if n_expert_demos is None:
        return demos
    else:
        if len(demos) < n_expert_demos:
            raise ValueError(
                f"Want to use n_expert_demos={n_expert_demos} trajectories, but only "
                f"{len(demos)} are available.",
            )
        if len(demos) > n_expert_demos:
            logger.warning(
                f"Using only the first {n_expert_demos} trajectories out of "
                f"{len(demos)} available.",
            )
            return demos[:n_expert_demos]
        else:
            return demos


@demonstrations_ingredient.capture
def _generate_expert_trajs(
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
        raise ValueError("n_expert_demos must be specified when generating demos.")

    logger.info(f"Generating {n_expert_demos} expert trajectories")
    with environment.make_rollout_venv() as rollout_env:
        return rollout.rollout(
            expert.get_expert_policy(rollout_env),
            rollout_env,
            rollout.make_sample_until(min_episodes=n_expert_demos),
            rng=_rnd,
        )


@demonstrations_ingredient.capture
def _download_expert_rollouts(
    environment: Dict[str, Any],
    path: Optional[str],
    organization: Optional[str],
    algo_name: Optional[str],
    loader_kwargs: Dict[str, Any],
):
    if path is not None:
        repo_id = path
    else:
        model_name = hfsb3.ModelName(
            algo_name,
            hfsb3.EnvironmentName(environment["gym_id"]),
        )
        repo_id = hfsb3.ModelRepoId(organization, model_name)

    logger.info(f"Loading expert trajectories from {repo_id}")
    dataset = datasets.load_dataset(repo_id, **loader_kwargs)
    return huggingface_utils.TrajectoryDatasetSequence(dataset)
