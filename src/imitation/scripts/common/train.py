"""Common configuration elements for training imitation algorithms."""

import logging
from typing import Any, Mapping, Union

import sacred
from stable_baselines3.common import base_class, policies, torch_layers, vec_env
from stable_baselines3.sac.policies import SACPolicy

import imitation.util.networks
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


@train_ingredient.named_config
def sac():
    policy_cls = SACPolicy

    locals()  # quieten flake8


@train_ingredient.named_config
def normalize_disable():
    policy_kwargs = {  # noqa: F841
        # FlattenExtractor is the default for SB3; but we specify it here
        # explicitly as no entry will be set to normalization by default
        # via the config hook.
        "features_extractor_class": torch_layers.FlattenExtractor,
    }


NORMALIZE_RUNNING_POLICY_KWARGS = {
    "features_extractor_class": base.NormalizeFeaturesExtractor,
    "features_extractor_kwargs": {
        "normalize_class": imitation.util.networks.RunningNorm,
    },
}


@train_ingredient.named_config
def normalize_running():
    policy_kwargs = NORMALIZE_RUNNING_POLICY_KWARGS  # noqa: F841


@train_ingredient.config_hook
def config_hook(config, command_name, logger):
    """Sets defaults equivalent to `normalize_running`."""
    del command_name, logger
    if "features_extractor_class" not in config["train"]["policy_kwargs"]:
        return {"policy_kwargs": NORMALIZE_RUNNING_POLICY_KWARGS}
    return {}


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


@train_ingredient.capture
def suppress_sacred_error(policy_kwargs: Mapping[str, Any]):
    """No-op so Sacred recognizes `policy_kwargs` is used (in `rl` and elsewhere)."""
