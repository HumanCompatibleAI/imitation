import gym
import pytest
import tensorflow as tf

from airl import AIRLTrainer
from reward_net import BasicRewardNet
import util


def _init_trainer(env):
    rn = BasicRewardNet(env)
    policy = util.make_blank_policy(env, init_tensorboard=False)
    obs_old, act, obs_new = util.generate_rollouts(policy, env, 100)
    trainer = AIRLTrainer(env, policy=policy,
            reward_net=rn, expert_obs_old=obs_old,
            expert_act=act, expert_obs_new=obs_new)
    return policy, trainer

class TestAIRL(tf.test.TestCase):

    def test_init_no_crash(self, env='CartPole-v1'):
        _init_trainer(env)

    @pytest.mark.parameterize("n_timesteps", [100, 20, 1])
    def test_train_disc_no_crash(self, env='CartPole-v1', n_timesteps=100):
        policy, trainer = _init_trainer(env)
        obs_old, act, obs_new = util.generate_rollouts(policy, env,
                n_timesteps)
        trainer.train_disc(trainer.expert_obs_old, trainer.expert_act,
                trainer.expert_obs_new, obs_old, act, obs_new)

    def test_train_gen_no_crash(self, env='CartPole-v1', n_steps=10):
        policy, trainer = _init_trainer(env)
        trainer.train_gen(n_steps)
