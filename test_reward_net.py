import pytest
import unittest

import gym
from reward_net import BasicRewardNet


class TestBasicRewardNet(unittest.TestCase):

    def test_init_no_crash(self):
        for i in range(3):
            x = BasicRewardNet(gym.make('FrozenLake-v0'))
