"""This ingredient performs evaluation of learned policy.

It takes care of the right wrappers, does some rollouts and computes statistics of the rollouts.
"""

from typing import Mapping, Union

import numpy as np
import sacred
from stable_baselines3.common import base_class, policies, vec_env

from imitation.data import rollout

policy_evaluation_ingredient = sacred.Ingredient("policy_evaluation")


@policy_evaluation_ingredient.config
def config():
    n_episodes_eval = 50  # Num of episodes for final mean ground truth return
    locals()  # quieten flake8


@policy_evaluation_ingredient.named_config
def fast():
    n_episodes_eval = 1  # noqa: F841


@policy_evaluation_ingredient.capture
def eval_policy(
    rl_algo: Union[base_class.BaseAlgorithm, policies.BasePolicy],
    venv: vec_env.VecEnv,
    n_episodes_eval: int,
    _rnd: np.random.Generator,
) -> Mapping[str, float]:
    """Evaluation of imitation learned policy.

    Has the side effect of setting `rl_algo`'s environment to `venv`
    if it is a `BaseAlgorithm`.

    Args:
        rl_algo: Algorithm to evaluate.
        venv: Environment to evaluate on.
        n_episodes_eval: The number of episodes to average over when calculating
            the average episode reward of the imitation policy for return.
        _rnd: Random number generator provided by Sacred.

    Returns:
        A dictionary with two keys. "imit_stats" gives the return value of
        `rollout_stats()` on rollouts test-reward-wrapped environment, using the final
        policy (remember that the ground-truth reward can be recovered from the
        "monitor_return" key). "expert_stats" gives the return value of
        `rollout_stats()` on the expert demonstrations loaded from `rollout_path`.
    """
    sample_until_eval = rollout.make_min_episodes(n_episodes_eval)
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
    trajs = rollout.generate_trajectories(
        rl_algo,
        train_env,
        sample_until=sample_until_eval,
        rng=_rnd,
    )
    return rollout.rollout_stats(trajs)
