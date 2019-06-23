import pytest
import tensorflow as tf

from imitation.main.reward_net import BasicRewardNet, BasicShapedRewardNet

ENVS = ['FrozenLake-v0', 'CartPole-v1', 'Pendulum-v0']


@pytest.mark.parametrize("env", ENVS)
def test_init_no_crash(env):
    for i in range(3):
        with tf.variable_scope(env + str(i) + "shaped"):
            BasicShapedRewardNet(env)
        with tf.variable_scope(env + str(i)):
            BasicRewardNet(env)
    tf.reset_default_graph()
