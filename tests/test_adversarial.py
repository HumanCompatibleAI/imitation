"""Tests for imitation.algorithms.adversarial"""

import os

import pytest

from imitation.algorithms import adversarial
from imitation.data import rollout, types
from imitation.util import logger, util

# List of (algorithm_cls, algorithm_kwargs).
ALGORITHM_PARAMS = [
    (adversarial.AIRL, {}),
    (adversarial.GAIL, {"discrim_net_kwargs": {"positive_rewards": True}}),
    (adversarial.GAIL, {"discrim_net_kwargs": {"positive_rewards": False}}),
]
IN_CODECOV = "COV_CORE_CONFIG" in os.environ
# Disable SubprocVecEnv tests for code coverage test since
# multiprocessing support is flaky in py.test --cov
PARALLEL = [False] if IN_CODECOV else [True, False]


@pytest.fixture(autouse=True)
def setup_and_teardown(session):
    # Uses conftest.session fixture for everything in this file.
    yield


@pytest.fixture(params=PARALLEL)
def _parallel(request):
    """Auto-parametrizes `_parallel` for the `model` fixture.

    This way we don't have to add a @pytest.mark.parametrize("_parallel", ... )
    decorator in front of every test. I couldn't find a better way to do this that
    didn't involve the aforementioned `parameterize` duplication."""
    return request.param


@pytest.fixture(params=ALGORITHM_PARAMS)
def _algorithm_params(request):
    """Auto-parametrizes `_algorithm_params` for the `model` fixture."""
    return request.param


@pytest.fixture
def model(_algorithm_params, _parallel: bool, tmpdir: str):
    algorithm_cls, algorithm_kwargs = _algorithm_params
    trajs = types.load("tests/data/expert_models/cartpole_0/rollouts/final.pkl")
    expert_demos = rollout.flatten_trajectories(trajs)
    logger.configure(tmpdir, ["tensorboard", "stdout"])

    venv = util.make_vec_env(
        "CartPole-v1", n_envs=2, parallel=_parallel, log_dir=tmpdir,
    )

    gen_policy = util.init_rl(venv, verbose=1)

    return algorithm_cls(
        venv=venv,
        expert_demos=expert_demos,
        gen_policy=gen_policy,
        log_dir=tmpdir,
        **algorithm_kwargs,
    )


def test_train_disc_step_no_crash(tmpdir, model, n_timesteps=200):
    transitions = rollout.generate_transitions(
        model.gen_policy, model.venv, n_timesteps=n_timesteps
    )
    model.train_disc_step(gen_samples=transitions)


def test_train_gen_train_disc_no_crash(tmpdir, model, n_updates=2):
    model.train_gen(n_updates * model.gen_batch_size)
    model.train_disc()


@pytest.mark.expensive
def test_train_disc_improve_D(tmpdir, model, n_timesteps=200, n_steps=1000):
    gen_samples = rollout.generate_transitions(
        model.gen_policy, model.venv_train_norm, n_timesteps=n_timesteps
    )
    loss1 = model.eval_disc_loss(gen_samples=gen_samples)
    for _ in range(n_steps):
        model.train_disc_step(gen_samples=gen_samples)
    loss2 = model.eval_disc_loss(gen_samples=gen_samples)
    assert loss2 < loss1
