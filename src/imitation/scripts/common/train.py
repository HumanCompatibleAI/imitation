"""Common configuration elements for training imitation algorithms."""

import logging
from typing import Mapping, Union

import sacred
from stable_baselines3.common import base_class, policies, vec_env

from imitation.data import rollout
from imitation.policies import base

train_ingredient = sacred.Ingredient("train")
logger = logging.getLogger(__name__)


@train_ingredient.config
def config():
    # Training
    policy_cls = base.FeedForward32Policy
    policy_kwargs = {}

    # Evaluation
    n_episodes_eval = 50  # Num of episodes for final mean ground truth return

    locals()  # quieten flake8


@train_ingredient.named_config
def fast():
    n_episodes_eval = 1
    locals()  # quieten flake8


@train_ingredient.capture
def eval_policy(
    rl_algo: Union[base_class.BaseAlgorithm, policies.BasePolicy],
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
