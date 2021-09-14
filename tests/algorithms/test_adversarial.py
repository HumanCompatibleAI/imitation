"""Tests for imitation.trainer.Trainer and util.trainer.init_trainer."""

import os

import pytest
import seals  # noqa: F401
import stable_baselines3
from stable_baselines3.common import policies
from torch.utils import data as th_data

from imitation.algorithms import adversarial
from imitation.data import rollout, types
from imitation.util import logger, util

ALGORITH_KWARGS = {
    "airl-ppo": {
        "algorithm_cls": adversarial.AIRL,
        "model_class": stable_baselines3.PPO,
        "policy_class": policies.ActorCriticPolicy,
    },
    "gail-ppo": {
        "algorithm_cls": adversarial.GAIL,
        "model_class": stable_baselines3.PPO,
        "policy_class": policies.ActorCriticPolicy,
    },
    "gail-dqn": {
        "algorithm_cls": adversarial.GAIL,
        "model_class": stable_baselines3.DQN,
        "policy_class": stable_baselines3.dqn.MlpPolicy,
    },
}
IN_CODECOV = "COV_CORE_CONFIG" in os.environ
# Disable SubprocVecEnv tests for code coverage test since
# multiprocessing support is flaky in py.test --cov
PARALLEL = [False] if IN_CODECOV else [True, False]


@pytest.fixture(params=ALGORITH_KWARGS.values(), ids=ALGORITH_KWARGS.keys())
def _algorithm_kwargs(request):
    """Auto-parametrizes `_rl_algorithm_cls` for the `trainer` fixture."""
    return dict(request.param)


def test_train_disc_small_expert_data_warning(
    tmpdir,
    custom_logger,
    _algorithm_kwargs,
):
    venv = util.make_vec_env(
        "seals/CartPole-v0",
        n_envs=1,
        parallel=_parallel,
    )

    _algorithm_cls = _algorithm_kwargs.pop("algorithm_cls")
    gen_algo = util.init_rl(venv, verbose=1, **_algorithm_kwargs)
    small_data = rollout.generate_transitions(gen_algo, venv, n_timesteps=20)

    with pytest.raises(ValueError, match="Transitions.*expert_batch_size"):
        _algorithm_cls(
            venv=venv,
            expert_data=small_data,
            expert_batch_size=21,
            gen_algo=gen_algo,
            log_dir=tmpdir,
            custom_logger=custom_logger,
        )

    with pytest.raises(ValueError, match="expert_batch_size.*positive"):
        _algorithm_cls(
            venv=venv,
            expert_data=small_data,
            expert_batch_size=-1,
            gen_algo=gen_algo,
            log_dir=tmpdir,
            custom_logger=custom_logger,
        )


def test_airl_fail_fast(custom_logger, tmpdir):
    venv = util.make_vec_env(
        "seals/CartPole-v0",
        n_envs=1,
        parallel=_parallel,
    )

    gen_algo = util.init_rl(
        venv,
        model_class=stable_baselines3.DQN,
        policy_class=stable_baselines3.dqn.MlpPolicy,
    )
    small_data = rollout.generate_transitions(gen_algo, venv, n_timesteps=20)

    with pytest.raises(TypeError, match="AIRL needs a stochastic policy.*"):
        adversarial.AIRL(
            venv=venv,
            expert_data=small_data,
            expert_batch_size=20,
            gen_algo=gen_algo,
            log_dir=tmpdir,
            custom_logger=custom_logger,
        )


@pytest.fixture(params=PARALLEL)
def _parallel(request):
    """Auto-parametrizes `_parallel` for the `trainer` fixture.

    This way we don't have to add a @pytest.mark.parametrize("_parallel", ... )
    decorator in front of every test. I couldn't find a better way to do this that
    didn't involve the aforementioned `parameterize` duplication."""
    return request.param


@pytest.fixture(params=[True, False])
def _convert_dataset(request):
    """Auto-parametrizes `_convert_dataset` for the `trainer` fixture."""
    return request.param


@pytest.fixture(params=[1, 128])
def expert_batch_size(request):
    return request.param


@pytest.fixture
def expert_transitions():
    trajs = types.load("tests/testdata/expert_models/cartpole_0/rollouts/final.pkl")
    trans = rollout.flatten_trajectories(trajs)
    return trans


@pytest.fixture
def trainer(
    _algorithm_kwargs,
    _parallel: bool,
    tmpdir: str,
    _convert_dataset: bool,
    expert_batch_size: int,
    expert_transitions: types.Transitions,
):
    if _convert_dataset:
        expert_data = th_data.DataLoader(
            expert_transitions,
            batch_size=expert_batch_size,
            collate_fn=types.transitions_collate_fn,
            shuffle=True,
            drop_last=True,
        )
    else:
        expert_data = expert_transitions

    venv = util.make_vec_env(
        "seals/CartPole-v0",
        n_envs=1,
        parallel=_parallel,
        log_dir=tmpdir,
    )

    _algorithm_cls = _algorithm_kwargs.pop("algorithm_cls")
    gen_algo = util.init_rl(venv, verbose=1, **_algorithm_kwargs)
    custom_logger = logger.configure(tmpdir, ["tensorboard", "stdout"])
    trainer = _algorithm_cls(
        venv=venv,
        expert_data=expert_data,
        expert_batch_size=expert_batch_size,
        gen_algo=gen_algo,
        log_dir=tmpdir,
        custom_logger=custom_logger,
    )

    try:
        yield trainer
    finally:
        venv.close()


def test_train_disc_no_samples_error(trainer: adversarial.AdversarialTrainer):
    with pytest.raises(RuntimeError, match="No generator samples"):
        trainer.train_disc()


def test_train_disc_unequal_expert_gen_samples_error(trainer, expert_transitions):
    """Test that train_disc raises error when n_gen != n_expert samples."""
    if len(expert_transitions) < 2:  # pragma: no cover
        raise ValueError("Test assumes at least 2 samples.")
    expert_samples = types.dataclass_quick_asdict(expert_transitions)
    gen_samples = types.dataclass_quick_asdict(expert_transitions[:-1])
    with pytest.raises(ValueError, match="n_expert"):
        trainer.train_disc(expert_samples=expert_samples, gen_samples=gen_samples)


def test_train_disc_step_no_crash(trainer, expert_batch_size):
    transitions = rollout.generate_transitions(
        trainer.gen_algo,
        trainer.venv,
        n_timesteps=expert_batch_size,
        truncate=True,
    )
    trainer.train_disc(gen_samples=types.dataclass_quick_asdict(transitions))


def test_train_gen_train_disc_no_crash(trainer, n_updates=2):
    trainer.train_gen(n_updates * trainer.gen_train_timesteps)
    trainer.train_disc()


@pytest.mark.expensive
def test_train_disc_improve_D(
    tmpdir,
    trainer,
    expert_transitions,
    expert_batch_size,
    n_steps=3,
):
    expert_samples = expert_transitions[:expert_batch_size]
    expert_samples = types.dataclass_quick_asdict(expert_samples)
    gen_samples = rollout.generate_transitions(
        trainer.gen_algo,
        trainer.venv_train,
        n_timesteps=expert_batch_size,
        truncate=True,
    )
    gen_samples = types.dataclass_quick_asdict(gen_samples)
    init_stats = final_stats = None
    for _ in range(n_steps):
        final_stats = trainer.train_disc(
            gen_samples=gen_samples,
            expert_samples=expert_samples,
        )
        if init_stats is None:
            init_stats = final_stats
    assert final_stats["disc_loss"] < init_stats["disc_loss"]
