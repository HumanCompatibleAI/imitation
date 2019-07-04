import pytest
import tensorflow as tf

from imitation.reward_net import BasicRewardNet, BasicShapedRewardNet

ENVS = ['FrozenLake-v0', 'CartPole-v1', 'Pendulum-v0']


@pytest.mark.parametrize("env", ENVS)
def test_init_no_crash(env):
  for i in range(3):
    with tf.variable_scope(env + str(i) + "shaped"):
      BasicShapedRewardNet(env.observation_space, env.action_space)
    with tf.variable_scope(env + str(i)):
      BasicRewardNet(env.observation_space, env.action_space)
  tf.reset_default_graph()
