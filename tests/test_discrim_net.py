import numpy as np
import pytest
import tensorflow as tf

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
def test_discrim_net_no_crash(session, env_name, discrim_net_cls):
    # If parallel=True, codecov sometimes acts up.
    venv = util.make_vec_env(env_name, parallel=False)
    DISCRIM_NET_SETUPS[discrim_net_cls](venv)


@pytest.mark.parametrize("env_name", ENVS)
@pytest.mark.parametrize("discrim_net_cls", DISCRIM_NETS)
def test_serialize_identity(session, env_name, discrim_net_cls, tmpdir):
    """Does output of deserialized discriminator match that of original?"""
    venv = util.make_vec_env(env_name, parallel=False)
    original = DISCRIM_NET_SETUPS[discrim_net_cls](venv)
    random = base.RandomPolicy(venv.observation_space, venv.action_space)
    session.run(tf.global_variables_initializer())

    original.save(tmpdir)
    with tf.variable_scope("loaded"):
        loaded = discrim_net.DiscrimNet.load(tmpdir)

    transitions = rollout.generate_transitions(random, venv, n_timesteps=100)
    length = len(transitions.obs)  # n_timesteps is only a lower bound
    labels = np.random.randint(2, size=length).astype(np.float32)
    log_prob = np.random.randn(length)

    feed_dict = {}
    outputs = {"train": [], "test": []}
    for net in [original, loaded]:
        feed_dict.update(
            {
                net.obs_ph: transitions.obs,
                net.act_ph: transitions.acts,
                net.next_obs_ph: transitions.next_obs,
                net.labels_gen_is_one_ph: labels,
                net.log_policy_act_prob_ph: log_prob,
            }
        )
        outputs["train"].append(net.policy_train_reward)
        outputs["test"].append(net.policy_test_reward)

    rewards = session.run(outputs, feed_dict=feed_dict)

    for key, predictions in rewards.items():
        assert len(predictions) == 2
        assert np.allclose(predictions[0], predictions[1])
