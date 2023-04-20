"""Code to evaluate trained policies."""
from __future__ import annotations

import dataclasses
import typing
from typing import Any, Mapping, Union

from hydra.utils import call
from omegaconf import MISSING

from imitation_cli.utils import environment as environment_cfg
from imitation_cli.utils import randomness

if typing.TYPE_CHECKING:
    from stable_baselines3.common import base_class, policies, vec_env


@dataclasses.dataclass
class Config:
    """Configuration for evaluating a policy."""

    environment: environment_cfg.Config = MISSING
    n_episodes_eval: int = 50
    rng: randomness.Config = randomness.Config()


def register_configs(group: str, defaults: Mapping[str, Any] = {}) -> None:
    from hydra.core.config_store import ConfigStore

    cs = ConfigStore.instance()
    cs.store(
        name="default_evaluation",
        group=group,
        node=Config(**defaults),
    )
    cs.store(
        name="fast_evaluation",
        group=group,
        node=Config(n_episodes_eval=2, **defaults),
    )


def eval_policy(
    rl_algo: Union[base_class.BaseAlgorithm, policies.BasePolicy],
    config: Config,
) -> typing.Mapping[str, float]:
    """Evaluation of imitation learned policy.

    Has the side effect of setting `rl_algo`'s environment to `venv`
    if it is a `BaseAlgorithm`.

    Args:
        rl_algo: Algorithm to evaluate.
        config: Configuration for evaluation.

    Returns:
        A dictionary with two keys. "imit_stats" gives the return value of
        `rollout_stats()` on rollouts test-reward-wrapped environment, using the final
        policy (remember that the ground-truth reward can be recovered from the
        "monitor_return" key). "expert_stats" gives the return value of
        `rollout_stats()` on the expert demonstrations loaded from `rollout_path`.
    """
    from stable_baselines3.common import base_class

    from imitation.data import rollout

    sample_until_eval = rollout.make_min_episodes(config.n_episodes_eval)
    venv = call(config.environment)
    rng = call(config.rng)

    if isinstance(rl_algo, base_class.BaseAlgorithm):
        # Set RL algorithm's env to venv, removing any cruft wrappers that the RL
        # algorithm's environment may have accumulated.
        rl_algo.set_env(venv)
        # Generate trajectories with the RL algorithm's env - SB3 may apply wrappers
        # under the hood to get it to work with the RL algorithm (e.g. transposing
        # images, so they can be fed into CNNs).
        train_env = rl_algo.get_env()
        assert train_env is not None
    else:
        train_env = venv

    train_env = typing.cast(vec_env.VecEnv, train_env)
    trajs = rollout.generate_trajectories(
        rl_algo,
        train_env,
        sample_until=sample_until_eval,
        rng=rng,
    )
    return rollout.rollout_stats(trajs)
