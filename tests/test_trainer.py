"""Tests for imitation.trainer.Trainer and util.trainer.init_trainer."""
from contextlib import nullcontext
import os
import pickle

import pytest

from imitation.algorithms.adversarial import init_trainer
from imitation.util import rollout
from imitation.util.buffering_wrapper import BufferingWrapper

USE_GAIL = [True, False]
IN_CODECOV = 'COV_CORE_CONFIG' in os.environ
# Disable SubprocVecEnv tests for code coverage test since
# multiprocessing support is flaky in py.test --cov
PARALLEL = [False] if IN_CODECOV else [True, False]


@pytest.fixture(autouse=True)
def setup_and_teardown(session):
  # Uses conftest.session fixture for everything in this file
  yield


def init_test_trainer(tmpdir: str,
                      use_gail: bool = True,
                      parallel: bool = False,
                      **kwargs):
  with open("tests/data/expert_models/cartpole_0/rollouts/final.pkl",
            "rb") as f:
    trajs = pickle.load(f)
  return init_trainer("CartPole-v1",
                      trajs,
                      log_dir=tmpdir,
                      use_gail=use_gail,
                      parallel=parallel,
                      **kwargs)


@pytest.mark.parametrize("use_gail", USE_GAIL)
@pytest.mark.parametrize("parallel", PARALLEL)
def test_init_no_crash(tmp_path, use_gail, parallel):
  init_test_trainer(tmp_path, use_gail=use_gail, parallel=parallel)


@pytest.mark.parametrize("use_gail", USE_GAIL)
@pytest.mark.parametrize("parallel", PARALLEL)
def test_train_disc_step_no_crash(tmpdir, use_gail, parallel, n_timesteps=200):
  trainer = init_test_trainer(tmpdir, use_gail=use_gail, parallel=parallel)
  transitions = rollout.generate_transitions(trainer.gen_policy,
                                             trainer.venv,
                                             n_timesteps=n_timesteps)
  trainer.train_disc_step(gen_samples=transitions)


@pytest.mark.parametrize("use_gail", USE_GAIL)
@pytest.mark.parametrize("parallel", PARALLEL)
def test_train_gen_train_disc_no_crash(tmpdir, use_gail, parallel, n_updates=2):
  trainer = init_test_trainer(
    tmpdir=tmpdir, use_gail=use_gail, parallel=parallel)
  trainer.train_gen(n_updates * trainer.gen_batch_size)
  trainer.train_disc()


@pytest.mark.expensive
@pytest.mark.parametrize("use_gail", USE_GAIL)
def test_train_disc_improve_D(tmpdir, use_gail, n_timesteps=200, n_steps=1000):
  trainer = init_test_trainer(tmpdir, use_gail)
  gen_samples = rollout.generate_transitions(trainer.gen_policy,
                                             trainer.venv_train_norm,
                                             n_timesteps=n_timesteps)
  loss1 = trainer.eval_disc_loss(gen_samples=gen_samples)
  for _ in range(n_steps):
    trainer.train_disc_step(gen_samples=gen_samples)
  loss2 = trainer.eval_disc_loss(gen_samples=gen_samples)
  assert loss2 < loss1


def test_error_on_unexpected_env_change(tmpdir):
  # No `error_on_unexpected_policy_change=False` case because leads to
  # other error.
  trainer = init_test_trainer(
    tmpdir=tmpdir,
    trainer_kwargs=dict(error_on_unexpected_policy_change=True))

  trainer.gen_policy.set_env(BufferingWrapper(trainer.venv_train_norm))
  with pytest.raises(ValueError, match="Unexpected change to .*"):
    trainer.train(1024)


@pytest.mark.parametrize("errors_enabled", [True, False])
def test_error_on_unexpected_env_reset(tmpdir, errors_enabled):
  if errors_enabled:
    ctx = pytest.raises(ValueError, match="Unexpected extra reset .*")
  else:
    ctx = nullcontext()

  trainer = init_test_trainer(
    tmpdir=tmpdir,
    trainer_kwargs=dict(error_on_unexpected_policy_change=errors_enabled))
  trainer.train(1024)
  trainer.venv_train_norm_buffering.reset()
  with ctx:
    trainer.train(1024)
