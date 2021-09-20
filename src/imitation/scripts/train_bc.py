"""Trains a policy via behavioral cloning (BC) from expert demonstrations."""

import logging
import os.path as osp
import pathlib
from typing import Any, Mapping, Optional, Type

import torch as th
from sacred.observers import FileStorageObserver
from stable_baselines3.common import policies, utils, vec_env

from imitation.algorithms import bc
from imitation.data import rollout
from imitation.scripts.common import train
from imitation.scripts.config.train_bc import train_bc_ex

logger = logging.getLogger(__name__)


@train_bc_ex.capture
def make_policy(
    venv: vec_env.VecEnv,
    policy_cls: Type[policies.BasePolicy],
    policy_kwargs: Mapping[str, Any],
) -> policies.BasePolicy:
    """Makes policy.

    Args:
        venv: Vectorized environment we will be imitating demos from.
        policy_cls: Type of a Stable Baselines3 policy architecture.
        policy_kwargs: Keyword arguments for policy constructor.

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
    policy = policy_cls(**policy_kwargs)
    logger.info(f"Policy network summary:\n {policy}")
    return policy


@train_bc_ex.main
def train_bc(
    _run,
    batch_size: int,
    n_epochs: Optional[int],
    n_batches: Optional[int],
    l2_weight: float,
    optimizer_cls: Type[th.optim.Optimizer],
    optimizer_kwargs: dict,
    log_interval: int,
    log_rollouts_n_episodes: int,
) -> Mapping[str, Mapping[str, float]]:
    """Sacred interface to Behavioral Cloning.

    Args:
        batch_size: Number of observation-action samples used in each BC update.
        n_epochs: The total number of training epochs. Set exactly one of n_epochs and
            n_batches.
        n_batches: The total number of training batches. Set exactly one of n_epochs and
            n_batches.
        l2_weight: L2 regularization weight.
        optimizer_cls: The Torch optimizer class used for BC updates.
        optimizer_kwargs: keyword arguments, excluding learning rate and
              weight decay, for optimiser construction.
        log_interval: The number of updates in between logging various training
            statistics to stdout and Tensorboard.
        log_rollouts_n_episodes: The number of rollout episodes generated for
            training statistics every `log_interval` updates. If `venv` is None or
            this argument is nonpositive, then no rollouts are generated.

    Returns:
        Statistics for rollouts from the trained policy and demonstration data.
    """
    custom_logger, log_dir = train.setup_logging()
    venv = train.make_venv()
    expert_trajs = train.load_expert_trajs()
    expert_transitions = rollout.flatten_trajectories(expert_trajs)
    logger.info(f"Loaded {len(expert_transitions)} timesteps of expert data")

    policy = make_policy(venv)
    model = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        policy=policy,
        demonstrations=expert_transitions,
        demo_batch_size=batch_size,
        l2_weight=l2_weight,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
        custom_logger=custom_logger,
    )
    # TODO(adam): have BC support timesteps instead of epochs/batches?
    model.train(
        n_epochs=n_epochs,
        n_batches=n_batches,
        log_interval=log_interval,
        log_rollouts_venv=venv,
        log_rollouts_n_episodes=log_rollouts_n_episodes,
    )
    # TODO(adam): add checkpointing to BC?
    model.save_policy(policy_path=pathlib.Path(log_dir, "final.th"))

    results = {}
    results["imit_stats"] = train.eval_policy(model.policy, venv)
    results["expert_stats"] = rollout.rollout_stats(expert_trajs)
    return results


def main_console():
    observer = FileStorageObserver(osp.join("output", "sacred", "train_bc"))
    train_bc_ex.observers.append(observer)
    train_bc_ex.run_commandline()


if __name__ == "__main__":
    main_console()
