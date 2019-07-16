"""Tests for imitation.trainer.Trainer and util.trainer.init_trainer."""
import os

import pytest

from imitation import util
from imitation.util import rollout
from imitation.util.trainer import init_trainer

USE_GAIL = [True, False]
IN_CODECOV = 'COV_CORE_CONFIG' in os.environ
if IN_CODECOV:  # multiprocessing breaks codecov, disable
  PARALLEL = [False]
else:
  PARALLEL = [True, False]


@pytest.fixture(autouse=True)
def setup_and_teardown(session):
  # Uses conftest.session fixture for everything in this file
  yield


@pytest.mark.parametrize("use_gail", USE_GAIL)
@pytest.mark.parametrize("parallel", PARALLEL)
def test_init_no_crash(use_gail, parallel, env='CartPole-v1'):
  init_trainer(env, use_gail=use_gail, parallel=parallel)


@pytest.mark.parametrize("use_gail", USE_GAIL)
@pytest.mark.parametrize("parallel", PARALLEL)
def test_train_disc_no_crash(use_gail, parallel,
                             env='CartPole-v1', n_timesteps=200):
  trainer = init_trainer(env, use_gail=use_gail, parallel=parallel)
  trainer.train_disc()
  obs_old, act, obs_new, _ = rollout.generate_transitions(
      trainer.gen_policy, env, n_timesteps=n_timesteps)
  trainer.train_disc(gen_old_obs=obs_old, gen_act=act,
                     gen_new_obs=obs_new)


@pytest.mark.parametrize("use_gail", USE_GAIL)
@pytest.mark.parametrize("parallel", PARALLEL)
def test_train_gen_no_crash(use_gail, parallel, env='CartPole-v1', n_steps=10):
  trainer = init_trainer(env, use_gail=use_gail, parallel=parallel)
  trainer.train_gen(n_steps)


@pytest.mark.expensive
@pytest.mark.parametrize("use_gail", USE_GAIL)
def test_train_disc_improve_D(use_gail, env='CartPole-v1', n_timesteps=200,
                              n_steps=1000):
  trainer = init_trainer(env, use_gail=use_gail)
  obs_old, act, obs_new, _ = rollout.generate_transitions(
      trainer.gen_policy, env, n_timesteps=n_timesteps)
  kwargs = dict(gen_old_obs=obs_old, gen_act=act, gen_new_obs=obs_new)
  loss1 = trainer.eval_disc_loss(**kwargs)
  trainer.train_disc(n_steps=n_steps, **kwargs)
  loss2 = trainer.eval_disc_loss(**kwargs)
  assert loss2 < loss1


@pytest.mark.expensive
@pytest.mark.xfail(reason="(AIRL) With random seeding, this test passed 36 "
                   "times out of 40.",
                   raises=AssertionError)
def test_train_gen_degrade_D(use_gail=False, env='CartPole-v1', n_timesteps=200,
                             n_steps=10000):
  trainer = init_trainer(env, use_gail=use_gail)
  if use_gail:
    kwargs = {}
  else:
    obs_old, act, obs_new, _ = rollout.generate_transitions(
        trainer.gen_policy, env, n_timesteps=n_timesteps)
    kwargs = dict(gen_old_obs=obs_old, gen_act=act, gen_new_obs=obs_new)

  loss1 = trainer.eval_disc_loss(**kwargs)
  trainer.train_gen(n_steps=n_steps)
  loss2 = trainer.eval_disc_loss(**kwargs)
  assert loss2 > loss1


@pytest.mark.expensive
@pytest.mark.xfail(reason="(AIRL) With random seeding, this test passed 19 "
                   "times out of 30.",
                   raises=AssertionError)
def test_train_disc_then_gen(use_gail=False, env='CartPole-v1', n_timesteps=200,
                             n_steps=10000):
  trainer = init_trainer(env, use_gail=use_gail)
  if use_gail:
    kwargs = {}
  else:
    obs_old, act, obs_new, _ = rollout.generate_transitions(
        trainer.gen_policy, env, n_timesteps=n_timesteps)
    kwargs = dict(gen_old_obs=obs_old, gen_act=act, gen_new_obs=obs_new)

  loss1 = trainer.eval_disc_loss(**kwargs)
  trainer.train_disc(n_steps=n_steps, **kwargs)
  loss2 = trainer.eval_disc_loss(**kwargs)
  trainer.train_gen(n_steps=n_steps)
  loss3 = trainer.eval_disc_loss(**kwargs)
  assert loss2 < loss1
  assert loss3 > loss2


@pytest.mark.expensive
@pytest.mark.parametrize("use_gail", USE_GAIL)
def test_train_no_crash(use_gail, env='CartPole-v1'):
  trainer = init_trainer(env, use_gail=use_gail)
  trainer.train(n_epochs=1)


@pytest.mark.expensive
@pytest.mark.xfail(
    reason="Either AIRL train is broken or not enough epochs."
    " Consider making a plot of episode reward over time to check.",
    raises=AssertionError)
@pytest.mark.skip
def test_trained_policy_better_than_random(use_gail, env='CartPole-v1',
                                           n_episodes=50):
  """
  Make sure that generator policy trained to mimick expert policy
  demonstrations) achieves higher reward than a random policy.

  In other words, perform a basic check on the imitation learning
  capabilities of AIRL and GAIL.
  """
  env = util.make_vec_env(env, 32)
  trainer = init_trainer(env, use_random_expert=True, use_gail=use_gail)
  expert_policy = util.load_policy(env, basedir="expert_models")
  random_policy = util.make_blank_policy(env)
  if expert_policy is None:
    pytest.fail("Couldn't load expert_policy!")

  trainer.train(n_epochs=200)

  # Idea: Plot n_epochs vs generator reward.
  for _ in range(4):
    expert_ret = rollout.mean_return(expert_policy, env, n_episodes=n_episodes)
    gen_ret = rollout.mean_return(trainer.gen_policy, env,
                                  n_episodes=n_episodes)
    random_ret = rollout.mean_return(random_policy, env, n_episodes=n_episodes)

    print("expert return:", expert_ret)
    print("generator return:", gen_ret)
    print("random return:", random_ret)
    assert expert_ret > random_ret
    assert gen_ret > random_ret


@pytest.mark.expensive
@pytest.mark.parametrize("use_gail", USE_GAIL)
def test_wrap_learned_reward_no_crash(use_gail, env="CartPole-v1"):
  """
  Briefly train with AIRL, and then used the learned reward to wrap
  a duplicate environment. Finally, use that learned reward to train
  a policy.
  """
  trainer = init_trainer(env, use_gail=use_gail)
  trainer.train(n_epochs=1)

  learned_reward_env = trainer.wrap_env_test_reward(env)
  policy = util.make_blank_policy(env)
  policy.set_env(learned_reward_env)
  policy.learn(10)
