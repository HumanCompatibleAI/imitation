import pytest
import tensorflow as tf

from yairl.reward_net import BasicRewardNet, BasicShapedRewardNet

ENVS = ['FrozenLake-v0', 'CartPole-v1', 'Pendulum-v0']


@pytest.mark.parametrize("env", ENVS)
def test_init_no_crash(env):
    for i in range(3):
        with tf.variable_scope(env+str(i)+"shaped"):
            BasicShapedRewardNet(env)
        with tf.variable_scope(env+str(i)):
            BasicRewardNet(env)
