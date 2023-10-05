"""Trains DAgger on synthetic demonstrations generated from an expert policy."""

import logging
import os.path as osp
import pathlib
from typing import Any, Dict, Mapping, Optional, Sequence, cast

import numpy as np
from sacred.observers import FileStorageObserver

from imitation.algorithms import dagger as dagger_algorithm
from imitation.algorithms import sqil as sqil_algorithm
from imitation.data import rollout, types
from imitation.scripts.config.train_imitation import train_imitation_ex
from imitation.scripts.ingredients import bc as bc_ingredient
from imitation.scripts.ingredients import demonstrations, environment, expert
from imitation.scripts.ingredients import logging as logging_ingredient
from imitation.scripts.ingredients import policy_evaluation
from imitation.util import util

logger = logging.getLogger(__name__)


def _all_trajectories_have_reward(trajectories: Sequence[types.Trajectory]) -> bool:
    """Returns True if all trajectories have reward information."""
    return all(isinstance(t, types.TrajectoryWithRew) for t in trajectories)


def _try_computing_expert_stats(
    expert_trajs: Sequence[types.Trajectory],
) -> Optional[Mapping[str, float]]:
    """Adds expert statistics to `stats` if all expert trajectories have reward."""
    if _all_trajectories_have_reward(expert_trajs):
        return rollout.rollout_stats(
            cast(Sequence[types.TrajectoryWithRew], expert_trajs),
        )
    else:
        logger.warning(
            "Expert trajectories do not have reward information, so expert "
            "statistics cannot be computed.",
        )
        return None


def _collect_stats(
    imit_stats: Mapping[str, float],
    expert_trajs: Sequence[types.Trajectory],
) -> Mapping[str, Mapping[str, Any]]:
    stats = {"imit_stats": imit_stats}
    expert_stats = _try_computing_expert_stats(expert_trajs)
    if expert_stats is not None:
        stats["expert_stats"] = expert_stats

    return stats


@train_imitation_ex.command
def bc(
    bc: Dict[str, Any],
    _run,
    _rnd: np.random.Generator,
) -> Mapping[str, Mapping[str, float]]:
    """Runs BC training.

    Args:
        bc: Configuration for BC training.
        _run: Sacred run object.
        _rnd: Random number generator provided by Sacred.

    Returns:
        Statistics for rollouts from the trained policy and demonstration data.
    """
    custom_logger, log_dir = logging_ingredient.setup_logging()

    expert_trajs = demonstrations.get_expert_trajectories()
    with environment.make_venv() as venv:
        bc_trainer = bc_ingredient.make_bc(venv, expert_trajs, custom_logger)

        bc_train_kwargs = dict(log_rollouts_venv=venv, **bc["train_kwargs"])
        if bc_train_kwargs["n_epochs"] is None and bc_train_kwargs["n_batches"] is None:
            bc_train_kwargs["n_batches"] = 50_000

        bc_trainer.train(**bc_train_kwargs)
        # TODO(adam): add checkpointing to BC?
        util.save_policy(bc_trainer.policy, policy_path=osp.join(log_dir, "final.th"))

        imit_stats = policy_evaluation.eval_policy(bc_trainer.policy, venv)

    stats = _collect_stats(imit_stats, expert_trajs)

    return stats


@train_imitation_ex.command
def dagger(
    bc: Dict[str, Any],
    dagger: Mapping[str, Any],
    _run,
    _rnd: np.random.Generator,
) -> Mapping[str, Mapping[str, float]]:
    """Runs DAgger training.

    Args:
        bc: Configuration for BC training.
        dagger: Arguments for DAgger training.
        _run: Sacred run object.
        _rnd: Random number generator provided by Sacred.

    Returns:
        Statistics for rollouts from the trained policy and demonstration data.
    """
    custom_logger, log_dir = logging_ingredient.setup_logging()

    expert_trajs: Optional[Sequence[types.Trajectory]] = None
    if dagger["use_offline_rollouts"]:
        expert_trajs = demonstrations.get_expert_trajectories()

    with environment.make_venv() as venv:
        bc_trainer = bc_ingredient.make_bc(venv, expert_trajs, custom_logger)

        bc_train_kwargs = dict(log_rollouts_venv=venv, **bc["train_kwargs"])
        if bc_train_kwargs["n_epochs"] is None and bc_train_kwargs["n_batches"] is None:
            bc_train_kwargs["n_epochs"] = 4

        expert_policy = expert.get_expert_policy(venv)

        dagger_trainer = dagger_algorithm.SimpleDAggerTrainer(
            venv=venv,
            scratch_dir=osp.join(log_dir, "scratch"),
            expert_trajs=expert_trajs,
            expert_policy=expert_policy,
            custom_logger=custom_logger,
            bc_trainer=bc_trainer,
            beta_schedule=dagger["beta_schedule"],
            rng=_rnd,
        )

        dagger_trainer.train(
            total_timesteps=int(dagger["total_timesteps"]),
            bc_train_kwargs=bc_train_kwargs,
        )
        # TODO(adam): add checkpointing to DAgger?
        save_locations = dagger_trainer.save_trainer()
        print(f"Model saved to {save_locations}")

        imit_stats = policy_evaluation.eval_policy(bc_trainer.policy, venv)

    assert dagger_trainer._all_demos is not None
    stats = _collect_stats(imit_stats, dagger_trainer._all_demos)

    return stats


@train_imitation_ex.command
def sqil(
    sqil: Mapping[str, Any],
    policy: Mapping[str, Any],
    rl: Mapping[str, Any],
    _run,
    _rnd: np.random.Generator,
) -> Mapping[str, Mapping[str, float]]:
    custom_logger, log_dir = logging_ingredient.setup_logging()
    expert_trajs = demonstrations.get_expert_trajectories()

    with environment.make_venv() as venv:
        sqil_trainer = sqil_algorithm.SQIL(
            venv=venv,
            demonstrations=expert_trajs,
            policy=policy["policy_cls"],
            custom_logger=custom_logger,
            rl_algo_class=rl["rl_cls"],
            rl_kwargs=rl["rl_kwargs"],
        )

        sqil_trainer.train(
            total_timesteps=int(sqil["total_timesteps"]),
            **sqil["train_kwargs"],
        )
        util.save_policy(sqil_trainer.policy, policy_path=osp.join(log_dir, "final.th"))

        imit_stats = policy_evaluation.eval_policy(sqil_trainer.policy, venv)

    stats = _collect_stats(imit_stats, expert_trajs)

    return stats


def main_console():
    observer_path = pathlib.Path.cwd() / "output" / "sacred" / "train_imitation"
    observer = FileStorageObserver(observer_path)
    train_imitation_ex.observers.append(observer)
    train_imitation_ex.run_commandline()


if __name__ == "__main__":
    main_console()
