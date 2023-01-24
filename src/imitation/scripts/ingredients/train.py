"""Common configuration elements for training imitation algorithms."""

import logging
from typing import Any, Mapping

import sacred
from stable_baselines3.common import policies

import imitation.util.networks
from imitation.policies import base
from imitation.scripts.ingredients import logging as logging_ingredient

train_ingredient = sacred.Ingredient(
    "train",
    ingredients=[logging_ingredient.logging_ingredient],
)
logger = logging.getLogger(__name__)


@train_ingredient.config
def config():
    # Training
    policy_cls = base.FeedForward32Policy
    policy_kwargs = {}

    locals()  # quieten flake8


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
def suppress_sacred_error(policy_kwargs: Mapping[str, Any]):
    """No-op so Sacred recognizes `policy_kwargs` is used (in `rl` and elsewhere)."""
