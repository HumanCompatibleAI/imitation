"""This ingredient provides a newly constructed stable-baselines3 policy."""

import logging
from typing import Any, Mapping, Type

import sacred
from stable_baselines3.common import policies, utils, vec_env
from stable_baselines3.sac import policies as SACPolicies


import imitation.util.networks
from imitation.policies import base
from imitation.scripts.ingredients import logging as logging_ingredient

policy_ingredient = sacred.Ingredient(
    "policy",
    ingredients=[logging_ingredient.logging_ingredient],
)
logger = logging.getLogger(__name__)


@policy_ingredient.config
def config():
    # Training
    policy_cls = base.FeedForward32Policy
    policy_kwargs = {}

    locals()  # quieten flake8


@policy_ingredient.named_config
def sac():
    policy_cls = base.SAC1024Policy  # noqa: F841

@policy_ingredient.named_config
def sac256():
    policy_cls = SACPolicies.SACPolicy


NORMALIZE_RUNNING_POLICY_KWARGS = {
    "features_extractor_class": base.NormalizeFeaturesExtractor,
    "features_extractor_kwargs": {
        "normalize_class": imitation.util.networks.RunningNorm,
    },
}


@policy_ingredient.named_config
def normalize_running():
    policy_kwargs = NORMALIZE_RUNNING_POLICY_KWARGS  # noqa: F841


# Default config for CNN Policies
@policy_ingredient.named_config
def cnn_policy():
    policy_cls = policies.ActorCriticCnnPolicy  # noqa: F841


@policy_ingredient.capture
def make_policy(
    venv: vec_env.VecEnv,
    policy_cls: Type[policies.BasePolicy],
    policy_kwargs: Mapping[str, Any],
) -> policies.BasePolicy:
    """Makes policy.

    Args:
        venv: Vectorized environment we will be imitating demos from.
        policy_cls: Type of a Stable Baselines3 policy architecture.
            Specify only if policy_path is not specified.
        policy_kwargs: Keyword arguments for policy constructor.
            Specify only if policy_path is not specified.

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
    policy: policies.BasePolicy
    policy = policy_cls(**policy_kwargs)
    logger.info(f"Policy network summary:\n {policy}")
    return policy
