import tempfile

import gym
import numpy as np
import pytest
import tensorflow as tf

from imitation.reward_net import BasicRewardNet, BasicShapedRewardNet
from imitation.util import rollout

ENVS = ['FrozenLake-v0', 'CartPole-v1', 'Pendulum-v0']
REWARD_NETS = [BasicRewardNet, BasicShapedRewardNet]


@pytest.mark.parametrize("env_id", ENVS)
@pytest.mark.parametrize("reward_net_cls", REWARD_NETS)
def test_init_no_crash(session, env_id, reward_net_cls):
  env = gym.make(env_id)
  for i in range(3):
    with tf.variable_scope(env_id + str(i) + "shaped"):
      reward_net_cls(env.observation_space, env.action_space)


def _make_feed_dict(reward_net, rollouts):
  old_obs, act, new_obs, _rew = rollouts
  return {
      reward_net.old_obs_ph: old_obs,
      reward_net.act_ph: act,
      reward_net.new_obs_ph: new_obs,
  }


@pytest.mark.parametrize("env_id", ENVS)
@pytest.mark.parametrize("reward_net_cls", REWARD_NETS)
def test_serialize_identity(session, env_id, reward_net_cls):
  """Does output of deserialized reward network match that of original?"""
  env = gym.make(env_id)
  with tf.variable_scope("original"):
    original = reward_net_cls(env.observation_space, env.action_space)
  random = rollout.RandomPolicy(env.observation_space, env.action_space)
  session.run(tf.global_variables_initializer())

  with tempfile.TemporaryDirectory(prefix='imitation-serialize') as tmpdir:
    original.save(tmpdir)
    with tf.variable_scope("loaded"):
      loaded = reward_net_cls.load(tmpdir)

  assert original.observation_space == loaded.observation_space
  assert original.action_space == loaded.action_space

  rollouts = rollout.generate_transitions(random, env, n_timesteps=100)
  feed_dict = {}
  outputs = {'train': [], 'test': []}
  for net in [original, loaded]:
    feed_dict.update(_make_feed_dict(net, rollouts))
    outputs['train'].append(net.reward_output_train)
    outputs['test'].append(net.reward_output_test)

  rewards = session.run(outputs, feed_dict=feed_dict)

  for key, predictions in rewards.items():
    assert len(predictions) == 2
    assert np.allclose(predictions[0], predictions[1])
