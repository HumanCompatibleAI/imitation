import pytest
import tensorflow as tf

from imitation import util
from imitation.util import rollout
from imitation.util.trainer import init_trainer

use_gail_vals = [True, False]


@pytest.fixture(autouse=True)
def setup_and_teardown():
  yield
  tf.reset_default_graph()


@pytest.mark.parametrize("use_gail", use_gail_vals)
def test_init_no_crash(use_gail, env='CartPole-v1'):
  init_trainer(env, use_gail=use_gail)


@pytest.mark.parametrize("use_gail", use_gail_vals)
def test_train_disc_no_crash(use_gail, env='CartPole-v1', n_timesteps=200):
  trainer = init_trainer(env, use_gail=use_gail)
  trainer.train_disc()
  obs_old, act, obs_new, _ = rollout.generate_transitions(
      trainer.gen_policy, env, n_timesteps=n_timesteps)
  trainer.train_disc(gen_old_obs=obs_old, gen_act=act,
                     gen_new_obs=obs_new)


@pytest.mark.parametrize("use_gail", use_gail_vals)
def test_train_gen_no_crash(use_gail, env='CartPole-v1', n_steps=10):
  trainer = init_trainer(env, use_gail=use_gail)
  trainer.train_gen(n_steps)


@pytest.mark.expensive
@pytest.mark.parametrize("use_gail", use_gail_vals)
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
@pytest.mark.parametrize("use_gail", use_gail_vals)
def test_train_no_crash(use_gail, env='CartPole-v1'):
  trainer = init_trainer(env, use_gail=use_gail)
  trainer.train(n_epochs=1)


@pytest.mark.expensive
@pytest.mark.parametrize("use_gail", use_gail_vals)
def test_wrap_learned_reward_no_crash(use_gail, env="CartPole-v1"):
  """
  Briefly train with AIRL, and then used the learned reward to wrap
  a duplicate environment. Finally, use that learned reward to train
  a policy.
  """
  trainer = init_trainer(env, use_gail=use_gail)
  trainer.train(n_epochs=1)

  learned_reward_env = trainer.wrap_env_test_reward(env)
  policy = util.make_blank_policy(env, init_tensorboard=False)
  policy.set_env(learned_reward_env)
  policy.learn(10)
