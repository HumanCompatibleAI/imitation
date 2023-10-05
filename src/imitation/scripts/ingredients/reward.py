"""This ingredient provides a reward network."""

import logging
import typing
from typing import Any, Mapping, Optional, Type

import sacred
from stable_baselines3.common import vec_env

from imitation.rewards import reward_nets
from imitation.util import networks

reward_ingredient = sacred.Ingredient("reward")
logger = logging.getLogger(__name__)


@reward_ingredient.config
def config():
    # Custom reward network
    net_cls = None
    net_kwargs = {}
    normalize_output_layer = networks.RunningNorm
    add_std_alpha = None
    ensemble_size = None
    ensemble_member_config = {}
    locals()  # quieten flake8


@reward_ingredient.named_config
def normalize_input_disable():
    net_kwargs = {"normalize_input_layer": None}  # noqa: F841


@reward_ingredient.named_config
def normalize_input_running():
    net_kwargs = {"normalize_input_layer": networks.RunningNorm}  # noqa: F841


@reward_ingredient.named_config
def normalize_output_disable():
    normalize_output_layer = None  # noqa: F841


@reward_ingredient.named_config
def normalize_output_running():
    normalize_output_layer = networks.RunningNorm  # noqa: F841


@reward_ingredient.named_config
def normalize_output_ema():
    normalize_output_layer = networks.EMANorm  # noqa: F841


@reward_ingredient.named_config
def reward_ensemble():
    net_cls = reward_nets.RewardEnsemble
    add_std_alpha = 0
    ensemble_size = 5
    normalize_output_layer = None
    ensemble_member_config = {
        "net_cls": reward_nets.BasicRewardNet,
        "net_kwargs": {},
        "normalize_output_layer": networks.RunningNorm,
    }
    locals()


@reward_ingredient.config_hook
def config_hook(config, command_name, logger):
    """Sets default values for `net_cls` and `net_kwargs`."""
    del logger
    res = {}
    if config["reward"]["net_cls"] is None:
        default_net = reward_nets.BasicRewardNet
        if command_name == "airl":
            default_net = reward_nets.BasicShapedRewardNet
        res["net_cls"] = default_net

    if "normalize_input_layer" not in config["reward"]["net_kwargs"]:
        res["net_kwargs"] = {"normalize_input_layer": networks.RunningNorm}

    if "net_cls" in res and issubclass(res["net_cls"], reward_nets.RewardEnsemble):
        del res["net_kwargs"]["normalize_input_layer"]

    return res


def _make_reward_net(
    venv: vec_env.VecEnv,
    net_cls: Type[reward_nets.RewardNet],
    net_kwargs: Mapping[str, Any],
    normalize_output_layer: Optional[Type[networks.BaseNorm]],
):
    """Helper function for creating reward nets."""
    reward_net = net_cls(
        venv.observation_space,
        venv.action_space,
        **net_kwargs,
    )

    if normalize_output_layer is not None:
        reward_net = reward_nets.NormalizedRewardNet(
            reward_net,
            normalize_output_layer,
        )

    return reward_net


@reward_ingredient.capture
def make_reward_net(
    venv: vec_env.VecEnv,
    net_cls: Type[reward_nets.RewardNet],
    net_kwargs: Mapping[str, Any],
    normalize_output_layer: Optional[Type[networks.BaseNorm]],
    add_std_alpha: Optional[float],
    ensemble_size: Optional[int],
    ensemble_member_config: Optional[Mapping[str, Any]],
) -> reward_nets.RewardNet:
    """Builds a reward network.

    Args:
        venv: Vectorized environment reward network will predict reward for.
        net_cls: Class of reward network to construct.
        net_kwargs: Keyword arguments passed to reward network constructor.
        normalize_output_layer: Wrapping the reward_net with NormalizedRewardNet
            to normalize the reward output.
        add_std_alpha: multiple of reward function standard deviation to add to the
            reward in predict_processed. Must be None when using a reward function that
            does not keep track of variance. Defaults to None.
        ensemble_size: The number of ensemble members to create. Must set if using
            `net_cls =` :class: `reward_nets.RewardEnsemble`.
        ensemble_member_config: The configuration for individual ensemble
            members. Note that `ensemble_member_config.net_cls` must not be
            :class: `reward_nets.RewardEnsemble`. Must be set if using
            `net_cls = ` :class: `reward_nets.RewardEnsemble`.

    Returns:
        A, possibly wrapped, instance of `net_cls`.

    Raises:
        ValueError: Using a reward ensemble but failed to provide configuration.
    """
    if issubclass(net_cls, reward_nets.RewardEnsemble):
        net_cls = typing.cast(Type[reward_nets.RewardEnsemble], net_cls)
        if ensemble_member_config is None:
            raise ValueError("Must specify ensemble_member_config.")

        if ensemble_size is None:
            raise ValueError("Must specify ensemble_size.")

        members = [
            _make_reward_net(venv, **ensemble_member_config)
            for _ in range(ensemble_size)
        ]

        reward_net: reward_nets.RewardNet = net_cls(
            venv.observation_space,
            venv.action_space,
            members,
        )

        if add_std_alpha is not None:
            assert isinstance(reward_net, reward_nets.RewardNetWithVariance)
            reward_net = reward_nets.AddSTDRewardWrapper(
                reward_net,
                default_alpha=add_std_alpha,
            )

        if normalize_output_layer is not None:
            raise ValueError("Output normalization not supported on RewardEnsembles.")

        return reward_net
    else:
        return _make_reward_net(venv, net_cls, net_kwargs, normalize_output_layer)
