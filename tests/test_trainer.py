"""Tests for imitation.trainer.Trainer and util.trainer.init_trainer."""
import os

import pytest

from imitation import util
from imitation.util import rollout
from imitation.util.trainer import init_trainer

USE_GAIL = [True, False]
IN_CODECOV = 'COV_CORE_CONFIG' in os.environ
# Disable SubprocVecEnv tests for code coverage test since
# multiprocessing support is flaky in py.test --cov
PARALLEL = [False] if IN_CODECOV else [True, False]


@pytest.fixture(autouse=True)
def setup_and_teardown(session):
  # Uses conftest.session fixture for everything in this file
  yield


def init_test_trainer(env_id: str, use_gail: bool, parallel: bool = False):
  return init_trainer(env_id=env_id,
                      rollout_glob=f"tests/data/rollouts/{env_id}*.pkl",
                      use_gail=use_gail,
                      parallel=parallel)


@pytest.mark.parametrize("use_gail", USE_GAIL)
@pytest.mark.parametrize("parallel", PARALLEL)
def test_init_no_crash(use_gail, parallel, env='CartPole-v1'):
  init_test_trainer(env, use_gail=use_gail, parallel=parallel)


@pytest.mark.parametrize("use_gail", USE_GAIL)
@pytest.mark.parametrize("parallel", PARALLEL)
def test_train_disc_no_crash(use_gail, parallel,
                             env='CartPole-v1', n_timesteps=200):
  trainer = init_test_trainer(env, use_gail=use_gail, parallel=parallel)
  trainer.train_disc()
  obs_old, act, obs_new, _ = rollout.generate_transitions(
      trainer.gen_policy, env, n_timesteps=n_timesteps)
  trainer.train_disc(gen_old_obs=obs_old, gen_act=act,
                     gen_new_obs=obs_new)


@pytest.mark.parametrize("use_gail", USE_GAIL)
@pytest.mark.parametrize("parallel", PARALLEL)
def test_train_gen_no_crash(use_gail, parallel, env='CartPole-v1', n_steps=10):
  trainer = init_test_trainer(env, use_gail=use_gail, parallel=parallel)
  trainer.train_gen(n_steps)


@pytest.mark.expensive
@pytest.mark.parametrize("use_gail", USE_GAIL)
def test_train_disc_improve_D(use_gail, env='CartPole-v1', n_timesteps=200,
                              n_steps=1000):
  trainer = init_test_trainer(env, use_gail)
  obs_old, act, obs_new, _ = rollout.generate_transitions(
      trainer.gen_policy, env, n_timesteps=n_timesteps)
  kwargs = dict(gen_old_obs=obs_old, gen_act=act, gen_new_obs=obs_new)
  loss1 = trainer.eval_disc_loss(**kwargs)
  trainer.train_disc(n_steps=n_steps, **kwargs)
  loss2 = trainer.eval_disc_loss(**kwargs)
  assert loss2 < loss1


@pytest.mark.expensive
@pytest.mark.parametrize("use_gail", USE_GAIL)
def test_train_no_crash(use_gail, env='CartPole-v1'):
  trainer = init_test_trainer(env, use_gail)
  trainer.train(n_epochs=1)


@pytest.mark.expensive
@pytest.mark.parametrize("use_gail", USE_GAIL)
def test_wrap_learned_reward_no_crash(use_gail, env="CartPole-v1"):
  """
  Briefly train with AIRL, and then used the learned reward to wrap
  a duplicate environment. Finally, use that learned reward to train
  a policy.
  """
  trainer = init_test_trainer(env, use_gail)
  trainer.train(n_epochs=1)

  learned_reward_env = trainer.wrap_env_test_reward(env)
  policy = util.init_rl(env)
  policy.set_env(learned_reward_env)
  policy.learn(10)
