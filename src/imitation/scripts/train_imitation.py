"""Trains DAgger on synthetic demonstrations generated from an expert policy."""

import logging
import os.path as osp
import pathlib
import warnings
from typing import Any, Mapping, Optional, Sequence, Type, cast

from sacred.observers import FileStorageObserver
from stable_baselines3.common import policies, utils, vec_env

from imitation.algorithms import bc as bc_algorithm
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.data import rollout, types
from imitation.scripts.common import common, demonstrations, expert, train
from imitation.scripts.config.train_imitation import train_imitation_ex

logger = logging.getLogger(__name__)


@train_imitation_ex.capture(prefix="train")
def make_policy(
    venv: vec_env.VecEnv,
    policy_cls: Type[policies.BasePolicy],
    policy_kwargs: Mapping[str, Any],
    agent_path: Optional[str],
) -> policies.BasePolicy:
    """Makes policy.

    Args:
        venv: Vectorized environment we will be imitating demos from.
        policy_cls: Type of a Stable Baselines3 policy architecture.
            Specify only if policy_path is not specified.
        policy_kwargs: Keyword arguments for policy constructor.
            Specify only if policy_path is not specified.
        agent_path: Path to serialized policy. If provided, then load the
            policy from this path. Otherwise, make a new policy.
            Specify only if policy_cls and policy_kwargs are not specified.

    Returns:
        A Stable Baselines3 policy.
    """
    policy_kwargs = dict(policy_kwargs)
    if issubclass(policy_cls, policies.ActorCriticPolicy):
        policy_kwargs.update(
            {
                "observation_space": venv.observation_space,
                "action_space": venv.action_space,
                # parameter mandatory for ActorCriticPolicy, but not used by BC
                "lr_schedule": utils.get_schedule_fn(1),
            },
        )
    policy: policies.BasePolicy
    if agent_path is not None:
        warnings.warn(
            "When agent_path is specified, policy_cls and policy_kwargs are ignored.",
            RuntimeWarning,
        )
        policy = bc_algorithm.reconstruct_policy(agent_path)
    else:
        policy = policy_cls(**policy_kwargs)
    logger.info(f"Policy network summary:\n {policy}")
    return policy


@train_imitation_ex.capture
def train_imitation(
    _run,
    bc_kwargs: Mapping[str, Any],
    bc_train_kwargs: Mapping[str, Any],
    dagger: Mapping[str, Any],
    use_dagger: bool,
    agent_path: Optional[str],
) -> Mapping[str, Mapping[str, float]]:
    """Runs DAgger (if `use_dagger`) or BC (otherwise) training.

    Args:
        bc_kwargs: Keyword arguments passed through to `bc.BC` constructor.
        bc_train_kwargs: Keyword arguments passed through to `BC.train()` method.
        dagger: Arguments for DAgger training.
        use_dagger: If True, train using DAgger; otherwise, use BC.
        agent_path: Path to serialized policy. If provided, then load the
            policy from this path. Otherwise, make a new policy.
            Specify only if policy_cls and policy_kwargs are not specified.

    Returns:
        Statistics for rollouts from the trained policy and demonstration data.
    """
    rng = common.make_rng()
    custom_logger, log_dir = common.setup_logging()

    with common.make_venv() as venv:
        imit_policy = make_policy(venv, agent_path=agent_path)

        expert_trajs: Optional[Sequence[types.Trajectory]] = None
        if not use_dagger or dagger["use_offline_rollouts"]:
            expert_trajs = demonstrations.get_expert_trajectories()

        bc_trainer = bc_algorithm.BC(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            policy=imit_policy,
            demonstrations=expert_trajs,
            custom_logger=custom_logger,
            rng=rng,
            **bc_kwargs,
        )
        bc_train_kwargs = dict(log_rollouts_venv=venv, **bc_train_kwargs)
        if bc_train_kwargs["n_epochs"] is None and bc_train_kwargs["n_batches"] is None:
            if use_dagger:
                bc_train_kwargs["n_epochs"] = 4
            else:
                bc_train_kwargs["n_batches"] = 50_000

        if use_dagger:
            expert_policy = expert.get_expert_policy(venv)
            model = SimpleDAggerTrainer(
                venv=venv,
                scratch_dir=osp.join(log_dir, "scratch"),
                expert_trajs=expert_trajs,
                expert_policy=expert_policy,
                custom_logger=custom_logger,
                bc_trainer=bc_trainer,
                rng=rng,
            )
            model.train(
                total_timesteps=int(dagger["total_timesteps"]),
                bc_train_kwargs=bc_train_kwargs,
            )
            # TODO(adam): add checkpointing to DAgger?
            save_locations = model.save_trainer()
            print(f"Model saved to {save_locations}")
        else:
            bc_trainer.train(**bc_train_kwargs)
            # TODO(adam): add checkpointing to BC?
            bc_trainer.save_policy(policy_path=osp.join(log_dir, "final.th"))

        imit_stats = train.eval_policy(imit_policy, venv)

    stats = {"imit_stats": imit_stats}
    trajectories = model._all_demos if use_dagger else expert_trajs
    assert trajectories is not None
    if all(isinstance(t, types.TrajectoryWithRew) for t in trajectories):
        expert_stats = rollout.rollout_stats(
            cast(Sequence[types.TrajectoryWithRew], trajectories),
        )
        stats["expert_stats"] = expert_stats
    return stats


@train_imitation_ex.command
def bc() -> Mapping[str, Mapping[str, float]]:
    """Run BC experiment using a Sacred interface to BC.

    Returns:
        Statistics for rollouts from the trained policy and expert data.
    """
    return train_imitation(use_dagger=False)


@train_imitation_ex.command
def dagger() -> Mapping[str, Mapping[str, float]]:
    """Run synthetic DAgger experiment using a Sacred interface to SimpleDAggerTrainer.

    Returns:
        Statistics for rollouts from the trained policy and expert data.
    """
    return train_imitation(use_dagger=True)


def main_console():
    observer_path = pathlib.Path.cwd() / "output" / "sacred" / "train_dagger"
    observer = FileStorageObserver(observer_path)
    train_imitation_ex.observers.append(observer)
    train_imitation_ex.run_commandline()


if __name__ == "__main__":
    main_console()
