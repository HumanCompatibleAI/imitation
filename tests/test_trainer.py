"""Tests for imitation.trainer.Trainer and util.trainer.init_trainer."""
import os
import pickle

import pytest

from imitation.algorithms.adversarial import init_trainer
from imitation.util import rollout

USE_GAIL = [True, False]
IN_CODECOV = 'COV_CORE_CONFIG' in os.environ
# Disable SubprocVecEnv tests for code coverage test since
# multiprocessing support is flaky in py.test --cov
PARALLEL = [False] if IN_CODECOV else [True, False]


@pytest.fixture(autouse=True)
def setup_and_teardown(session):
  # Uses conftest.session fixture for everything in this file
  yield


def init_test_trainer(tmpdir: str, use_gail: bool, parallel: bool = False):
  with open("tests/data/expert_models/cartpole_0/rollouts/final.pkl",
            "rb") as f:
    trajs = pickle.load(f)
  return init_trainer("CartPole-v1",
                      trajs,
                      log_dir=tmpdir,
                      use_gail=use_gail,
                      parallel=parallel)


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
