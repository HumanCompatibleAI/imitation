"""This ingredient provides BC algorithm instance by either loading it from disk or constructing it from scratch."""
import warnings
from typing import Optional, Sequence

import sacred
import torch as th
from stable_baselines3.common import vec_env

from imitation.algorithms import bc
from imitation.data import types
from imitation.scripts.ingredients import policy

bc_ingredient = sacred.Ingredient("bc", ingredients=[policy.policy_ingredient])


@bc_ingredient.config
def config():
    batch_size = 32
    l2_weight = 3e-5  # L2 regularization weight
    optimizer_cls = th.optim.Adam
    optimizer_kwargs = dict(
        lr=4e-4,
    )
    train_kwargs = dict(
        n_epochs=None,  # Number of BC epochs per DAgger training round
        n_batches=None,  # Number of BC batches per DAgger training round
        log_interval=500,  # Number of updates between Tensorboard/stdout logs
    )
    agent_path = None  # Path to serialized policy. If None, a new policy is created.

    locals()  # quieten flake8 unused variable warning


@bc_ingredient.capture
def make_bc(
    venv: vec_env.VecEnv,
    expert_trajs: Sequence[types.Trajectory],
    custom_logger,
    batch_size: int,
    l2_weight: float,
    optimizer_cls,
    optimizer_kwargs,
    _rnd,
) -> bc.BC:
    return bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        policy=make_or_load_policy(venv),
        demonstrations=expert_trajs,
        custom_logger=custom_logger,
        rng=_rnd,
        batch_size=batch_size,
        l2_weight=l2_weight,
        optimizer_cls=optimizer_cls,
        optimizer_kwargs=optimizer_kwargs,
    )


@bc_ingredient.capture
def make_or_load_policy(venv: vec_env.VecEnv, agent_path: Optional[str]):
    """Makes a policy or loads a policy from a path if provided.

    Args:
        venv: Vectorized environment we will be imitating demos from.
        agent_path: Path to serialized policy. If provided, then load the
            policy from this path. Otherwise, make a new policy.
            Specify only if policy_cls and policy_kwargs are not specified.

    Returns:
        A Stable Baselines3 policy.
    """
    if agent_path is None:
        policy.make_policy(venv)
    else:
        warnings.warn(
            "When agent_path is specified, policy.policy_cls and policy.policy_kwargs "
            "are ignored.",
            RuntimeWarning,
        )
        return bc.reconstruct_policy(agent_path)
