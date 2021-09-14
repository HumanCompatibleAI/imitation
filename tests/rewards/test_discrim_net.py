import contextlib
import os

import numpy as np
import pytest
import torch as th
from stable_baselines3.common import vec_env

from imitation.data import rollout
from imitation.policies import base
from imitation.rewards import discrim_nets, reward_nets
from imitation.util import util


def _setup_airl_basic(venv):
    reward_net = reward_nets.BasicRewardNet(venv.observation_space, venv.action_space)
    return discrim_nets.DiscrimNetAIRL(reward_net)


def _setup_airl_basic_custom_net(venv):
    reward_net = reward_nets.BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        use_state=True,
        use_action=True,
        use_next_state=False,
        use_done=False,
        hid_sizes=(32, 32),
    )
    return discrim_nets.DiscrimNetAIRL(reward_net)


def _setup_airl_undiscounted_shaped_reward_net(venv):
    reward_net = reward_nets.BasicShapedRewardNet(
        venv.observation_space,
        venv.action_space,
        discount_factor=1.0,
        use_next_state=True,
        use_done=True,
        potential_hid_sizes=(32, 32),
    )
    return discrim_nets.DiscrimNetAIRL(reward_net)


def _setup_gail(venv):
    return discrim_nets.DiscrimNetGAIL(venv.observation_space, venv.action_space)


def _setup_gail_provide_discriminator(venv):
    discriminator = discrim_nets.ActObsMLP(
        venv.action_space,
        venv.observation_space,
        hid_sizes=(4, 4, 4),
    )
    return discrim_nets.DiscrimNetGAIL(
        venv.observation_space,
        venv.action_space,
        discriminator,
    )


DISCRIM_NET_SETUPS = {
    "AIRL_basic_reward_net": _setup_airl_basic,
    "AIRL_basic_reward_net_custom_base_net": _setup_airl_basic_custom_net,
    "AIRL_unshaped_reward_net_undiscounted": _setup_airl_undiscounted_shaped_reward_net,
    "GAIL": _setup_gail,
    "GAIL_custom_network": _setup_gail_provide_discriminator,
}

ENV_NAMES = ["FrozenLake-v1", "CartPole-v1", "Pendulum-v0"]


@pytest.fixture(params=ENV_NAMES)
def env_name(request):
    return request.param


@pytest.fixture
def venv(env_name):
    return util.make_vec_env(env_name, parallel=False)


@pytest.fixture(params=list(DISCRIM_NET_SETUPS.keys()))
def discrim_net(request, venv):
    """Fixture for every DiscrimNet in DISCRIM_NET_SETUPS.

    The `params` argument of the fixture decorator is over the keys of
    DISCRIM_NET_SETUPS rather than the values so that the tests have nice
    paramerized string names.
    """
    # If parallel=True, codecov sometimes acts up.
    return DISCRIM_NET_SETUPS[request.param](venv)


@pytest.mark.parametrize("n_timesteps", [2, 4, 10])
def test_logits_gen_is_high_log_policy_act_prob(
    discrim_net: discrim_nets.DiscrimNet,
    venv: vec_env.VecEnv,
    n_timesteps: int,
):
    """Smoke test calling `logits_gen_is_high` on each DiscrimNet.

    For subclasses of DiscrimNetAIRL, also checks that the function raises
    error on `log_policy_act_prob=None`.
    """

    random = base.RandomPolicy(venv.observation_space, venv.action_space)
    trans = rollout.generate_transitions(random, venv, n_timesteps=n_timesteps)

    obs = util.torchify_with_space(trans.obs, venv.observation_space)
    acts = util.torchify_with_space(trans.acts, venv.action_space)
    next_obs = util.torchify_with_space(trans.next_obs, venv.observation_space)
    dones = th.as_tensor(trans.dones)
    log_act_prob_non_none = th.as_tensor(np.random.rand(n_timesteps))

    for log_act_prob in [None, log_act_prob_non_none]:
        if (
            isinstance(discrim_net, discrim_nets.DiscrimNetAIRL)
            and log_act_prob is None
        ):
            maybe_error_ctx = pytest.raises(TypeError, match="Non-None.*required.*")
        else:
            maybe_error_ctx = contextlib.nullcontext()

        with maybe_error_ctx:
            discrim_net.logits_gen_is_high(obs, acts, next_obs, dones, log_act_prob)


def test_serialize_identity(discrim_net, venv, tmpdir):
    """Does output of deserialized discriminator match that of original?"""
    original = discrim_net
    random = base.RandomPolicy(venv.observation_space, venv.action_space)

    tmppath = os.path.join(tmpdir, "discrim_net.pt")
    th.save(original, tmppath)
    loaded = th.load(tmppath)

    transitions = rollout.generate_transitions(random, venv, n_timesteps=100)

    rewards = {"train": [], "test": []}
    for net in [original, loaded]:
        rewards["train"].append(
            net.predict_reward_train(
                transitions.obs,
                transitions.acts,
                transitions.next_obs,
                transitions.dones,
            ),
        )
        rewards["test"].append(
            net.predict_reward_test(
                transitions.obs,
                transitions.acts,
                transitions.next_obs,
                transitions.dones,
            ),
        )

    for key, predictions in rewards.items():
        assert len(predictions) == 2
        assert np.allclose(predictions[0], predictions[1])
