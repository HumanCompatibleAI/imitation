"""Common configuration elements for training imitation algorithms."""

import logging
from typing import Any, Mapping, Union

import sacred
from stable_baselines3.common import base_class, policies, vec_env

import imitation.util.networks
from imitation.data import rollout
from imitation.policies import base
from imitation.scripts.common import common

train_ingredient = sacred.Ingredient("train", ingredients=[common.common_ingredient])
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
    n_episodes_eval = 1  # noqa: F841


@train_ingredient.named_config
def sac():
    policy_cls = base.SAC1024Policy  # noqa: F841


NORMALIZE_RUNNING_POLICY_KWARGS = {
    "features_extractor_class": base.NormalizeFeaturesExtractor,
    "features_extractor_kwargs": {
        "normalize_class": imitation.util.networks.RunningNorm,
    },
}


@train_ingredient.named_config
def normalize_running():
    policy_kwargs = NORMALIZE_RUNNING_POLICY_KWARGS  # noqa: F841


# Default config for CNN Policies
@train_ingredient.named_config
def cnn_policy():
    policy_cls = policies.ActorCriticCnnPolicy  # noqa: F841


@train_ingredient.capture
def eval_policy(
    rl_algo: Union[base_class.BaseAlgorithm, policies.BasePolicy],
    venv: vec_env.VecEnv,
    n_episodes_eval: int,
) -> Mapping[str, float]:
    """Evaluation of imitation learned policy.

    Has the side effect of setting `rl_algo`'s environment to `venv`
    if it is a `BaseAlgorithm`.

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
    rng = common.make_rng()
    sample_until_eval = rollout.make_min_episodes(n_episodes_eval)
    if isinstance(rl_algo, base_class.BaseAlgorithm):
        # Set RL algorithm's env to venv, removing any cruft wrappers that the RL
        # algorithm's environment may have accumulated.
        rl_algo.set_env(venv)
        # Generate trajectories with the RL algorithm's env - SB3 may apply wrappers
        # under the hood to get it to work with the RL algorithm (e.g. transposing
        # images so they can be fed into CNNs).
        train_env = rl_algo.get_env()
        assert train_env is not None
    else:
        train_env = venv
    trajs = rollout.generate_trajectories(
        rl_algo,
        train_env,
        sample_until=sample_until_eval,
        rng=rng,
    )
    return rollout.rollout_stats(trajs)


@train_ingredient.capture
def suppress_sacred_error(policy_kwargs: Mapping[str, Any]):
    """No-op so Sacred recognizes `policy_kwargs` is used (in `rl` and elsewhere)."""
