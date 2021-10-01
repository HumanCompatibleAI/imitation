"""Trains DAgger on synthetic demonstrations generated from an expert policy."""

import logging
import os.path as osp
from typing import Any, Mapping, Optional, Type

from sacred.observers import FileStorageObserver
from stable_baselines3.common import policies, utils, vec_env

from imitation.algorithms.bc import BC
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.data import rollout
from imitation.policies import serialize
from imitation.scripts.common import common, demonstrations, train
from imitation.scripts.config.train_imitation import train_imitation_ex

logger = logging.getLogger(__name__)


@train_imitation_ex.capture(prefix="train")
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


@train_imitation_ex.capture(prefix="dagger")
def load_expert_policy(
    venv: vec_env.VecEnv,
    expert_policy_type: Optional[str],
    expert_policy_path: Optional[str],
) -> policies.BasePolicy:
    """Loads expert policy from `expert_policy_path`.

    Args:
        venv: Vectorized environment the policy will be operating in.
        expert_policy_type: Either 'ppo', 'zero', or 'random'. This is used as the
            `policy_type` argument to `imitation.policies.serialize.load_policy`.
        expert_policy_path: Either a path to a policy directory containing model.zip
            (and optionally, vec_normalize.pkl) or None if policy type is 'zero' or
            'random'. This is used as the `policy_path` argument to
            `imitation.policies.serialize.load_policy`.

    Returns:
        The deserialized expert policy from `expert_policy_path`.

    Raises:
        ValueError: `expert_policy_path` is None.
        TypeError: The policy loaded from `expert_policy_path` is not a SB3 policy.
    """
    if expert_policy_path is None:
        raise ValueError("expert_policy_path cannot be None")
    # TODO(shwang): Add support for directly loading a BasePolicy `*.th` file.
    expert_policy = serialize.load_policy(expert_policy_type, expert_policy_path, venv)
    if not isinstance(expert_policy, policies.BasePolicy):
        raise TypeError(f"Unexpected type for expert_policy: {type(expert_policy)}")
    return expert_policy


@train_imitation_ex.capture
def train_imitation(
    _run,
    bc_kwargs: Mapping[str, Any],
    bc_train_kwargs: Mapping[str, Any],
    dagger: Mapping[str, Any],
    use_dagger: bool,
) -> Mapping[str, Mapping[str, float]]:
    """Runs DAgger (if `use_dagger`) or BC (otherwise) training.

    Args:
        bc_kwargs: Keyword arguments passed through to `bc.BC` constructor.
        bc_train_kwargs: Keyword arguments passed through to `BC.train` method.
        dagger: Arguments for DAgger training.
        use_dagger: If True, train using DAgger; otherwise, use BC.

    Returns:
        Statistics for rollouts from the trained policy and demonstration data.
    """
    custom_logger, log_dir = common.setup_logging()
    venv = common.make_venv()
    imit_policy = make_policy(venv)

    expert_trajs = None
    if not use_dagger or dagger["use_offline_rollouts"]:
        expert_trajs = demonstrations.load_expert_trajs()

    bc_trainer = BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        policy=imit_policy,
        demonstrations=expert_trajs,
        custom_logger=custom_logger,
        **bc_kwargs,
    )
    bc_train_kwargs = dict(log_rollouts_venv=venv, **bc_train_kwargs)
    if bc_train_kwargs["n_epochs"] is None and bc_train_kwargs["n_batches"] is None:
        if use_dagger:
            bc_train_kwargs["n_epochs"] = 4
        else:
            bc_train_kwargs["n_batches"] = 50_000

    if use_dagger:
        expert_policy = load_expert_policy(venv=venv)
        model = SimpleDAggerTrainer(
            venv=venv,
            scratch_dir=osp.join(log_dir, "scratch"),
            expert_trajs=expert_trajs,
            expert_policy=expert_policy,
            custom_logger=custom_logger,
            bc_trainer=bc_trainer,
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

    results = {}
    results["imit_stats"] = train.eval_policy(imit_policy, venv)
    results["expert_stats"] = rollout.rollout_stats(
        model._all_demos if use_dagger else expert_trajs,
    )
    return results


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
    observer = FileStorageObserver(osp.join("output", "sacred", "train_dagger"))
    train_imitation_ex.observers.append(observer)
    train_imitation_ex.run_commandline()


if __name__ == "__main__":
    main_console()
