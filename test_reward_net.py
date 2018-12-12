import pytest
import unittest

import gym
from reward_net import BasicRewardNet


class TestBasicRewardNet(unittest.TestCase):

    @pytest.mark.parameterize("env", ['FrozenLake-v0', 'Cartpole-v1',
        'CarRacing-v0', 'LunarLander-v2'])
    def test_init_no_crash(self, env='FrozenLake-v0'):
        for i in range(3):
            x = BasicRewardNet(env)
