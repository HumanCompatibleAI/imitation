"""Tests for imitation.scripts.ingredients.reward."""
from typing import Any, Mapping

import pytest

from imitation.rewards import reward_nets
from imitation.scripts.ingredients import reward
from imitation.util import networks


@pytest.fixture
def member_config() -> Mapping[str, Any]:
    return {
        "net_cls": reward_nets.BasicRewardNet,
        "net_kwargs": {},
        "normalize_output_layer": None,
    }


def test_make_reward_ensemble(member_config, cartpole_venv):
    reward_net = reward.make_reward_net(
        venv=cartpole_venv,
        net_cls=reward_nets.RewardEnsemble,
        add_std_alpha=None,
        net_kwargs={},
        normalize_output_layer=None,
        ensemble_size=3,
        ensemble_member_config=member_config,
    )
    assert isinstance(reward_net, reward_nets.RewardEnsemble)

    reward_net = reward.make_reward_net(
        venv=cartpole_venv,
        net_cls=reward_nets.RewardEnsemble,
        add_std_alpha=0,
        net_kwargs={},
        normalize_output_layer=None,
        ensemble_size=3,
        ensemble_member_config=member_config,
    )
    assert isinstance(reward_net, reward_nets.AddSTDRewardWrapper)


def test_make_reward_errors(member_config, cartpole_venv):
    with pytest.raises(ValueError, match=r"Must specify ensemble_size."):
        reward.make_reward_net(
            venv=cartpole_venv,
            net_cls=reward_nets.RewardEnsemble,
            add_std_alpha=None,
            net_kwargs={},
            normalize_output_layer=None,
            ensemble_size=None,
            ensemble_member_config=member_config,
        )

    with pytest.raises(ValueError, match=r"Must specify ensemble_member_config."):
        reward.make_reward_net(
            venv=cartpole_venv,
            net_cls=reward_nets.RewardEnsemble,
            add_std_alpha=None,
            net_kwargs={},
            normalize_output_layer=None,
            ensemble_size=5,
            ensemble_member_config=None,
        )

    with pytest.raises(
        ValueError,
        match=r"Output normalization not supported on RewardEnsembles.",
    ):
        reward.make_reward_net(
            venv=cartpole_venv,
            net_cls=reward_nets.RewardEnsemble,
            add_std_alpha=None,
            net_kwargs={},
            normalize_output_layer=networks.RunningNorm,
            ensemble_size=5,
            ensemble_member_config=member_config,
        )
