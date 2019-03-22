import pytest
import tensorflow as tf

from yairl.reward_net import BasicShapedRewardNet, BasicRewardNet


# class TestBasicRewardNet(tf.test.TestCase):
@pytest.mark.parametrize("env", ['FrozenLake-v0', 'CartPole-v1', 'Pendulum-v0'])
# 'CarRacing-v0', 'LunarLander-v2']) #  I can't even intiialize these envs!
def test_init_no_crash(env):
    for i in range(3):
        with tf.variable_scope(env + str(i) + "shaped"):
            x = BasicShapedRewardNet(env)
        with tf.variable_scope(env + str(i)):
            x = BasicRewardNet(env)
