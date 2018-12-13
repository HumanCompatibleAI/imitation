import gym
import tensorflow as tf

from airl import AIRLTrainer
from reward_net import BasicRewardNet
import util


class TestAIRL(tf.test.TestCase):

    def test_init(self, env='CartPole-v1'):
            rn = BasicRewardNet(env)
            policy = util.make_blank_policy(env)
            obs_old, act, obs_new = util.generate_rollouts(policy, env, 100)
            trainer = AIRLTrainer(env, policy=policy,
                    reward_net=rn, expert_obs_old=obs_old,
                    expert_act=act, expert_obs_new=obs_new)
