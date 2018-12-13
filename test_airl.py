import pytest
import unittest

import gym

from airl import AIRLTrainer
from reward_net import BasicRewardNet
import util


class TestAIRL(unittest.TestCase):

    def test_init(self, env='CartPole-v1'):
        rn = BasicRewardNet(env)
        policy = util.make_blank_policy(env)
        roll_obs, roll_act = util.generate_rollouts(policy, 100)
        trainer = AIRLTrainer(env, policy=policy,
                reward_net=rn, expert_rollout_obs=roll_obs,
                expert_rollout_act=roll_act)
