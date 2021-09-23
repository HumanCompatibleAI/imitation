"""Trains DAgger on synthetic demonstrations generated from an expert policy."""

import logging
import os.path as osp
import pathlib
from typing import Mapping, Optional, Sequence, Type, Union

import torch as th
from sacred.observers import FileStorageObserver
from stable_baselines3.common import policies, vec_env

from imitation.algorithms import dagger
from imitation.data import rollout, types
from imitation.policies import serialize
from imitation.scripts.config.train_dagger import train_dagger_ex
from imitation.util import logger
from imitation.util import sacred as sacred_util


@train_dagger_ex.main
def train_dagger(
    _run,
    expert_data_src: Union[types.AnyPath, Sequence[types.Trajectory]],
    expert_data_src_format: str,
    n_expert_demos: Optional[int],
    expert_policy_path: types.AnyPath,
    expert_policy_type: str,
    venv: vec_env.VecEnv,
    total_timesteps: float,
    batch_size: int,
    bc_train_kwargs: dict,
    l2_weight: float,
    optimizer_cls: Type[th.optim.Optimizer],
    optimizer_kwargs: dict,
    log_dir: types.AnyPath,
    n_episodes_eval: int,
) -> Mapping[str, Mapping[str, float]]:
    """Run synthetic DAgger experiment using a Sacred interface to SimpleDAggerTrainer.

    Args:
        expert_data_src: Either a path to pickled `Sequence[Trajectory]` or
            `Sequence[Trajectory]` or None. If None, then ignore
            `expert_data_src_format` and `n_expert_demos`, and don't load
            initial demonstrations.
        expert_data_src_format: Either "path" if `expert_data_src` is a path, or
            "trajectory" if `expert_data_src` is `Sequence[Trajectory]`, or None.
            If None, then ignore `expert_data_src` and `n_expert_demos` and don't load
            initial demonstrations.
        n_expert_demos: If not None, then a positive number used to truncate the number
            expert demonstrations used from `expert_data_src`. If this number is larger
            than the total number of demonstrations available, then a ValueError is
            raised.
        expert_policy_path: Either a path to a policy directory containing model.zip
            (and optionally, vec_normalize.pkl) if `expert_data_src_format == "ppo"`,
            or None if `expert_data_src_format in ('zero', 'random'). This is used
            as the `policy_path` argument to `imitation.policies.serialize.load_policy`.
        expert_policy_type: Either 'ppo', 'zero', or 'random'. This is used as the
            `policy_type` argument to `imitation.policies.serialize.load_policy`.
        venv: The vectorized training environment matching the expert data and policy.
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
        log_dir: Log output directory. Final policy is also saved in this directory as
            "{log_dir}/final.pkl"
        n_episodes_eval: The number of final evaluation rollout episodes, if `venv` is
            provided. These rollouts are used to generate final statistics saved into
            Sacred results, which can be compiled into a table by
            `imitation.scripts.analyze.analyze_imitation`.

    Returns:
        Statistics for rollouts from the trained policy and expert data.

    Raises:
        ValueError: `expert_policy_path` is None.
        ValueError:  `expert_data_src_format` unrecognized.
        TypeError: `expert_policy` is not a BasePolicy.
    """
    # TODO(shwang): Add support for directly loading a BasePolicy `*.th` file.
    log_dir = pathlib.Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Logging to %s", log_dir)

    custom_logger = logger.configure(log_dir, ["tensorboard", "stdout"])
    sacred_util.build_sacred_symlink(log_dir, _run)

    if expert_policy_path is None:
        raise ValueError("expert_policy_path cannot be None")

    expert_policy = serialize.load_policy(expert_policy_type, expert_policy_path, venv)
    if not isinstance(expert_policy, policies.BasePolicy):
        raise TypeError(f"Unexpected type for expert_policy: {type(expert_policy)}")

    expert_trajs: Optional[Sequence[types.Trajectory]]
    if expert_data_src_format is None or expert_data_src is None:
        expert_trajs = None
    elif expert_data_src_format == "path":
        expert_trajs = types.load(expert_data_src)
    elif expert_data_src_format == "trajectory":
        # Convenience option for launching experiment from Python script with
        # in-memory trajectories.
        expert_trajs = expert_data_src
    else:
        raise ValueError(
            f"expert_data_src_format={expert_data_src_format} should be 'path', "
            "'trajectory' or None.",
        )

    if expert_trajs is not None:
        for x in expert_trajs:
            if not isinstance(x, types.Trajectory):
                raise ValueError(f"Expert data is the wrong type: {type(x)}")

        # TODO(shwang): Copied from scripts/train_adversarial -- refactor with "auto"?
        if n_expert_demos is not None:
            if not len(expert_trajs) >= n_expert_demos:
                raise ValueError(
                    f"Want to use n_expert_demos={n_expert_demos} trajectories, but "
                    f"only {len(expert_trajs)} are available.",
                )
            expert_trajs = expert_trajs[:n_expert_demos]

    model = dagger.SimpleDAggerTrainer(
        venv=venv,
        scratch_dir=log_dir / "scratch",
        expert_trajs=expert_trajs,
        expert_policy=expert_policy,
        batch_size=batch_size,
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
    print(f"Tensorboard command: tbl '{log_dir}'")

    sample_until = rollout.make_sample_until(
        min_timesteps=None,
        min_episodes=n_episodes_eval,
    )
    trajs = rollout.generate_trajectories(
        model.policy,
        venv,
        sample_until=sample_until,
    )
    results = {}
    results["expert_stats"] = rollout.rollout_stats(model._all_demos)
    results["imit_stats"] = rollout.rollout_stats(trajs)
    return results


def main_console():
    observer = FileStorageObserver(osp.join("output", "sacred", "train_dagger"))
    train_dagger_ex.observers.append(observer)
    train_dagger_ex.run_commandline()


if __name__ == "__main__":
    main_console()
