"""Common configuration elements for training imitation algorithms."""

import logging
import os
from typing import Mapping, Optional, Sequence

import sacred
from stable_baselines3.common import base_class, vec_env

from imitation.data import rollout, types
from imitation.policies import base

train_ingredient = sacred.Ingredient("train")
logger = logging.getLogger(__name__)


@train_ingredient.config
def config():
    # TODO(adam): common logging?
    env_name = "seals/CartPole-v0"  # environment to train on
    data_dir = "data/"
    rollout_path = None  # path to file containing rollouts
    n_expert_demos = None  # Num demos used. None uses every demo possible
    n_episodes_eval = 50  # Num of episodes for final mean ground truth return

    # TODO(adam): does this need to be here or could it be in rl and separately for bc?
    # Is there any script we want to not take a policy?
    # Hmm, MCE IRL doesn't need this -- it can just ignore it though.
    # (Perhaps have that script warn if something non-default is set...?)
    policy_cls = base.FeedForward32Policy
    policy_kwargs = {}

    # TODO(adam): should we separate config into a separate file?
    # or just disable F841 on this whole file?
    _ = locals()  # quiten flake8
    del _


@train_ingredient.config
def defaults(data_dir, env_name, rollout_path):
    # If rollout_path not set explicitly, then guess it based on environment name.
    if rollout_path is None:
        rollout_hint = env_name.split("-")[0].lower().replace("/", "_")
        rollout_path = os.path.join(
            data_dir,
            "expert_models",
            f"{rollout_hint}_0",
            "rollouts",
            "final.pkl",
        )
        del rollout_hint


@train_ingredient.named_config
def fast():
    n_expert_demos = 1
    n_episodes_eval = 1
    _ = locals()  # quiten flake8
    del _


@train_ingredient.capture
def load_expert_demos(
    rollout_path: str,
    n_expert_demos: Optional[int],
) -> Sequence[types.Trajectory]:
    """Loads expert demonstrations.

    Args:
        rollout_path: A path containing a pickled sequence of `types.Trajectory`.
        n_expert_demos: The number of trajectories to load.
            Dataset is truncated to this length if specified.

    Returns:
        The expert trajectories.

    Raises:
        ValueError: There are fewer trajectories than `n_expert_demos`.
    """
    expert_trajs = types.load(rollout_path)
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


@train_ingredient.capture
def eval_policy(
    rl_algo: base_class.BaseAlgorithm,
    venv: vec_env.VecEnv,
    n_episodes_eval: int,
) -> Mapping[str, float]:
    """Evaluation of imitation learned policy.

    Args:
        rl_algo: Algorithm to evaluate.
        venv: Environment to evaluate on.
        n_episodes_eval: The number of episodes to average over when calculating
            the average episode reward of the imitation policy for return.

    Returns:
        A dictionary with two keys. "imit_stats" gives the return value of
        `rollout_stats()` on rollouts test-reward-wrapped environment, using the final
        policy (remember that the ground-truth reward can be recovered from the
        "monitor_return" key). "expert_stats" gives the return value of
        `rollout_stats()` on the expert demonstrations loaded from `rollout_path`.

    """
    sample_until_eval = rollout.make_min_episodes(n_episodes_eval)
    trajs = rollout.generate_trajectories(
        rl_algo,
        venv,
        sample_until=sample_until_eval,
    )
    return rollout.rollout_stats(trajs)
