import numbers
import tempfile

import gym
import numpy as np
import pytest
import tensorflow as tf

from imitation.policies import base
from imitation.rewards import serialize
from imitation.util import rollout, util

ENVS = ['FrozenLake-v0', 'CartPole-v1', 'Pendulum-v0']
HARDCODED_TYPES = ['zero']


@pytest.mark.parametrize("env_id", ENVS)
@pytest.mark.parametrize("reward_net_cls", serialize.REWARD_NETS.values())
def test_init_no_crash(session, env_id, reward_net_cls):
  env = gym.make(env_id)
  for i in range(3):
    with tf.variable_scope(env_id + str(i) + "shaped"):
      reward_net_cls(env.observation_space, env.action_space)


def _sample(space, n):
  return np.array([space.sample() for _ in range(n)])


@pytest.mark.parametrize("env_name", ENVS)
@pytest.mark.parametrize("reward_type", HARDCODED_TYPES)
def test_reward_valid(env_name, reward_type):
  """Test output of reward function is appropriate shape and type."""
  venv = util.make_vec_env(env_name, n_envs=1, parallel=False)
  reward_fn = serialize.load_reward(reward_type, "foobar", venv)

  TRAJECTORY_LEN = 10
  old_obs = _sample(venv.observation_space, TRAJECTORY_LEN)
  actions = _sample(venv.action_space, TRAJECTORY_LEN)
  new_obs = _sample(venv.observation_space, TRAJECTORY_LEN)
  steps = np.arange(0, TRAJECTORY_LEN)

  pred_reward = reward_fn(old_obs, actions, new_obs, steps)
  assert pred_reward.shape == (TRAJECTORY_LEN, )
  assert isinstance(pred_reward[0], numbers.Number)


def _make_feed_dict(reward_net, rollouts):
  old_obs, act, new_obs, _rew = rollouts
  return {
      reward_net.old_obs_ph: old_obs,
      reward_net.act_ph: act,
      reward_net.new_obs_ph: new_obs,
  }


@pytest.mark.parametrize("env_name", ENVS)
@pytest.mark.parametrize("reward_net", serialize.REWARD_NETS.items())
def test_serialize_identity(session, env_name, reward_net):
  """Does output of deserialized reward network match that of original?"""
  net_name, net_cls = reward_net
  print(f"Testing {net_name}")

  venv = util.make_vec_env(env_name, n_envs=1, parallel=False)
  with tf.variable_scope("original"):
    original = net_cls(venv.observation_space, venv.action_space)
  random = base.RandomPolicy(venv.observation_space, venv.action_space)
  session.run(tf.global_variables_initializer())

  with tempfile.TemporaryDirectory(prefix='imitation-serialize-rew') as tmpdir:
    original.save(tmpdir)
    with tf.variable_scope("loaded"):
      loaded = net_cls.load(tmpdir)

    unshaped_fn = serialize.load_reward(f"{net_name}_unshaped", tmpdir, venv)
    shaped_fn = serialize.load_reward(f"{net_name}_shaped", tmpdir, venv)

  assert original.observation_space == loaded.observation_space
  assert original.action_space == loaded.action_space

  rollouts = rollout.generate_transitions(random, venv, n_timesteps=100)
  feed_dict = {}
  outputs = {'train': [], 'test': []}
  for net in [original, loaded]:
    feed_dict.update(_make_feed_dict(net, rollouts))
    outputs['train'].append(net.reward_output_train)
    outputs['test'].append(net.reward_output_test)

  rewards = session.run(outputs, feed_dict=feed_dict)

  old_obs, actions, new_obs, _ = rollouts
  steps = np.zeros((old_obs.shape[0], ))
  rewards['train'].append(shaped_fn(old_obs, actions, new_obs, steps))
  rewards['test'].append(unshaped_fn(old_obs, actions, new_obs, steps))

  for key, predictions in rewards.items():
    assert len(predictions) == 3
    assert np.allclose(predictions[0], predictions[1])
    assert np.allclose(predictions[0], predictions[2])
