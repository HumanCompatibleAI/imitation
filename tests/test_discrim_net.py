import os

import numpy as np
import pytest
import torch as th

from imitation.data import rollout
from imitation.policies import base
from imitation.rewards import discrim_net
from imitation.rewards.reward_net import BasicRewardNet
from imitation.util import util

ENVS = ["FrozenLake-v0", "CartPole-v1", "Pendulum-v0"]
DISCRIM_NETS = [discrim_net.DiscrimNetAIRL, discrim_net.DiscrimNetGAIL]


def _setup_airl(venv):
    reward_net = BasicRewardNet(venv.observation_space, venv.action_space)
    return discrim_net.DiscrimNetAIRL(reward_net)


def _setup_gail(venv):
    return discrim_net.DiscrimNetGAIL(venv.observation_space, venv.action_space)


DISCRIM_NET_SETUPS = {
    discrim_net.DiscrimNetAIRL: _setup_airl,
    discrim_net.DiscrimNetGAIL: _setup_gail,
}


@pytest.mark.parametrize("env_name", ENVS)
@pytest.mark.parametrize("discrim_net_cls", DISCRIM_NETS)
def test_discrim_net_no_crash(env_name, discrim_net_cls):
    # If parallel=True, codecov sometimes acts up.
    venv = util.make_vec_env(env_name, parallel=False)
    DISCRIM_NET_SETUPS[discrim_net_cls](venv)


@pytest.mark.parametrize("env_name", ENVS)
@pytest.mark.parametrize("discrim_net_cls", DISCRIM_NETS)
def test_serialize_identity(env_name, discrim_net_cls, tmpdir):
    """Does output of deserialized discriminator match that of original?"""
    venv = util.make_vec_env(env_name, parallel=False)
    original = DISCRIM_NET_SETUPS[discrim_net_cls](venv)
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
            )
        )
        rewards["test"].append(
            net.predict_reward_test(
                transitions.obs,
                transitions.acts,
                transitions.next_obs,
                transitions.dones,
            )
        )

    for key, predictions in rewards.items():
        assert len(predictions) == 2
        assert np.allclose(predictions[0], predictions[1])
