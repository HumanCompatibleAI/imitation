import tempfile

import gym
import numpy as np
import pytest
import tensorflow as tf

from imitation.discrim_net import DiscrimNetAIRL, DiscrimNetGAIL
from imitation.reward_net import BasicRewardNet
from imitation.util import rollout

ENVS = ['FrozenLake-v0', 'CartPole-v1', 'Pendulum-v0']
DISCRIM_NETS = [DiscrimNetAIRL, DiscrimNetGAIL]


def _setup_airl(env):
  reward_net = BasicRewardNet(env.observation_space, env.action_space)
  return DiscrimNetAIRL(reward_net)


def _setup_gail(env):
  return DiscrimNetGAIL(env.observation_space, env.action_space)


DISCRIM_NET_SETUPS = {
    DiscrimNetAIRL: _setup_airl,
    DiscrimNetGAIL: _setup_gail,
}


@pytest.mark.parametrize("env_id", ENVS)
@pytest.mark.parametrize("discrim_net_cls", DISCRIM_NETS)
def test_discrim_net_no_crash(session, env_id, discrim_net_cls):
  env = gym.make(env_id)
  DISCRIM_NET_SETUPS[discrim_net_cls](env)


@pytest.mark.parametrize("env_id", ENVS)
@pytest.mark.parametrize("discrim_net_cls", DISCRIM_NETS)
def test_serialize_identity(session, env_id, discrim_net_cls):
  """Does output of deserialized discriminator match that of original?"""
  env = gym.make(env_id)
  original = DISCRIM_NET_SETUPS[discrim_net_cls](env)
  random = rollout.RandomPolicy(env.observation_space, env.action_space)
  session.run(tf.global_variables_initializer())

  with tempfile.TemporaryDirectory(prefix='imitation-serialize') as tmpdir:
    original.save(tmpdir)
    with tf.variable_scope("loaded"):
      loaded = discrim_net_cls.load(tmpdir)

  old_obs, act, new_obs, _rew = rollout.generate_transitions(random, env,
                                                             n_timesteps=100)
  labels = np.random.randint(2, size=len(old_obs)).astype(np.float32)
  log_prob = np.random.randn(len(old_obs))

  feed_dict = {}
  outputs = {'train': [], 'test': []}
  for net in [original, loaded]:
    feed_dict.update({
        net.old_obs_ph: old_obs,
        net.act_ph: act,
        net.new_obs_ph: new_obs,
        net.labels_ph: labels,
        net.log_policy_act_prob_ph: log_prob,
    })
    outputs['train'].append(net.policy_train_reward)
    outputs['test'].append(net.policy_test_reward)

  rewards = session.run(outputs, feed_dict=feed_dict)

  for key, predictions in rewards.items():
    assert len(predictions) == 2
    assert np.allclose(predictions[0], predictions[1])
