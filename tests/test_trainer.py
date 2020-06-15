"""Tests for imitation.trainer.Trainer and util.trainer.init_trainer."""

import os

import pytest

from imitation.algorithms import adversarial
from imitation.data import dataset, rollout, types
from imitation.util import logger, util

ALGORITHM_CLS = [adversarial.AIRL, adversarial.GAIL]
IN_CODECOV = "COV_CORE_CONFIG" in os.environ
# Disable SubprocVecEnv tests for code coverage test since
# multiprocessing support is flaky in py.test --cov
PARALLEL = [False] if IN_CODECOV else [True, False]


@pytest.fixture(autouse=True)
def setup_and_teardown(session):
    # Uses conftest.session fixture for everything in this file
    yield


@pytest.fixture(params=PARALLEL)
def _parallel(request):
    # Auto-parametrizes `_parallel` for the trainer fixture. This way we don't have to
    # add a @pytest.mark.parametrize("_parallel", ... ) decorator in front of
    # every test.
    return request.param


@pytest.fixture(params=ALGORITHM_CLS)
def _algorithm_cls(request):
    # Auto-parametrizes `_algorithm_cls` for the trainer fixture.
    return request.param


# TODO(shwang): Consider renaming AdversarialTrainer to just Adversarial?
@pytest.fixture
def trainer(_algorithm_cls, _parallel: bool, tmpdir: str):
    trajs = types.load("tests/data/expert_models/cartpole_0/rollouts/final.pkl")
    expert_trans = rollout.flatten_trajectories(trajs)
    expert_dataset = dataset.SimpleDataset(data_map=dataclasses.asdict(expert_trans))
    logger.configure(tmpdir, ["tensorboard", "stdout"])

    venv = util.make_vec_env(
        "CartPole-v1",
        n_envs=2,
        parallel=_parallel,
        log_dir=tmpdir,
    )

    gen_policy = util.init_rl(venv, verbose=1)

    return _algorithm_cls(
        venv=venv,
        expert_dataset=expert_dataset,
        gen_policy=gen_policy,
        log_dir=tmpdir,
    )


def test_init_trainer_no_crash(trainer):
    """Smoke tests `trainer` fixture which initializes GAIL or AIRL."""

@pytest.mark.parametrize("use_gail", USE_GAIL)
@pytest.mark.parametrize("parallel", PARALLEL)
def test_train_disc_step_no_crash(tmpdir, use_gail, parallel, n_timesteps=200):
    trainer = init_test_trainer(tmpdir, use_gail=use_gail, parallel=parallel)
    transitions = rollout.generate_transitions(
        trainer.gen_policy, trainer.venv, n_timesteps=n_timesteps
    )
    trainer.train_disc_step(gen_samples=transitions)


@pytest.mark.parametrize("use_gail", USE_GAIL)
@pytest.mark.parametrize("parallel", PARALLEL)
def test_train_gen_train_disc_no_crash(tmpdir, use_gail, parallel, n_updates=2):
    trainer = init_test_trainer(tmpdir=tmpdir, use_gail=use_gail, parallel=parallel)
    trainer.train_gen(n_updates * trainer.gen_batch_size)
    trainer.train_disc()


@pytest.mark.expensive
@pytest.mark.parametrize("use_gail", USE_GAIL)
def test_train_disc_improve_D(tmpdir, use_gail, n_timesteps=200, n_steps=1000):
    trainer = init_test_trainer(tmpdir, use_gail)
    gen_samples = rollout.generate_transitions(
        trainer.gen_policy, trainer.venv_train_norm, n_timesteps=n_timesteps
    )
    loss1 = trainer.eval_disc_loss(gen_samples=gen_samples)
    for _ in range(n_steps):
        trainer.train_disc_step(gen_samples=gen_samples)
    loss2 = trainer.eval_disc_loss(gen_samples=gen_samples)
    assert loss2 < loss1
