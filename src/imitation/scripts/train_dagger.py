"""Trains DAgger on synthetic demonstrations generated from an expert policy."""

import os.path as osp
from typing import Mapping, Type

import torch as th
from sacred.observers import FileStorageObserver
from stable_baselines3.common import policies

from imitation.algorithms import dagger
from imitation.data import rollout, types
from imitation.policies import serialize
from imitation.scripts.common import train
from imitation.scripts.config.train_dagger import train_dagger_ex


@train_dagger_ex.main
def train_dagger(
    _run,
    use_offline_rollouts: bool,
    expert_policy_path: types.AnyPath,
    expert_policy_type: str,
    total_timesteps: float,
    batch_size: int,
    bc_train_kwargs: dict,
    l2_weight: float,
    optimizer_cls: Type[th.optim.Optimizer],
    optimizer_kwargs: dict,
) -> Mapping[str, Mapping[str, float]]:
    """Run synthetic DAgger experiment using a Sacred interface to SimpleDAggerTrainer.

    Args:
        use_offline_rollouts: If True, load pre-collected demonstrations to
            warm-start the policy with BC.
        expert_policy_path: Either a path to a policy directory containing model.zip
            (and optionally, vec_normalize.pkl) if `expert_data_src_format == "ppo"`,
            or None if `expert_data_src_format in ('zero', 'random'). This is used
            as the `policy_path` argument to `imitation.policies.serialize.load_policy`.
        expert_policy_type: Either 'ppo', 'zero', or 'random'. This is used as the
            `policy_type` argument to `imitation.policies.serialize.load_policy`.
        total_timesteps: The number of timesteps to train inside the environment. (In
            practice this is a lower bound, as the number of timesteps is rounded up
            to finish a DAgger training rounds.)
        batch_size: Number of observation-action samples used in each BC update.
        bc_train_kwargs: The `bc_train_kwargs` argument for
            `SimpleDAggerTrainer.train()`. A dict of keyword arguments that are passed
            to `BC.train()` every DAgger training round.
        l2_weight: L2 regularization weight for BC.
        optimizer_cls: The Torch optimizer class used for BC updates.
        optimizer_kwargs: Optimizer kwargs passed to BC.

    Returns:
        Statistics for rollouts from the trained policy and expert data.

    Raises:
        ValueError: `expert_policy_path` is None.
        TypeError: The policy loaded from `expert_policy_path` is not a SB3 policy.
    """
    custom_logger, log_dir = train.setup_logging()

    if expert_policy_path is None:
        raise ValueError("expert_policy_path cannot be None")

    # TODO(shwang): Add support for directly loading a BasePolicy `*.th` file.
    # set log_dir=None as saving trainer blows up if venv includes file handles
    # TODO(adam): should we even be saving venv when we save trainer?
    # (In general environments may not be pickleable.)
    venv = train.make_venv(log_dir=None)
    expert_policy = serialize.load_policy(expert_policy_type, expert_policy_path, venv)
    if not isinstance(expert_policy, policies.BasePolicy):
        raise TypeError(f"Unexpected type for expert_policy: {type(expert_policy)}")

    expert_trajs = None
    if use_offline_rollouts:
        expert_trajs = train.load_expert_trajs()

    model = dagger.SimpleDAggerTrainer(
        venv=venv,
        scratch_dir=osp.join(log_dir, "scratch"),
        expert_trajs=expert_trajs,
        expert_policy=expert_policy,
        batch_size=batch_size,
        # TODO(adam): this is only a subset of bc_kwargs... make this configurable?
        bc_kwargs=dict(
            l2_weight=l2_weight,
            optimizer_cls=optimizer_cls,
            optimizer_kwargs=optimizer_kwargs,
        ),
        custom_logger=custom_logger,
    )
    model.train(
        total_timesteps=int(total_timesteps),
        bc_train_kwargs=dict(
            log_rollouts_venv=venv,
            **bc_train_kwargs,
        ),
    )
    save_info_dict = model.save_trainer()
    print(f"Model saved to {save_info_dict}")

    results = {}
    results["expert_stats"] = rollout.rollout_stats(model._all_demos)
    results["imit_stats"] = train.eval_policy(model.policy, venv)
    return results


def main_console():
    observer = FileStorageObserver(osp.join("output", "sacred", "train_dagger"))
    train_dagger_ex.observers.append(observer)
    train_dagger_ex.run_commandline()


if __name__ == "__main__":
    main_console()
