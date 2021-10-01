"""Tests `imitation.algorithms.bc`."""

import dataclasses
import os

import pytest
import torch as th
from stable_baselines3.common import vec_env
from torch.utils import data as th_data

from imitation.algorithms import bc
from imitation.data import rollout, types
from imitation.util import logger, util

ROLLOUT_PATH = "tests/testdata/expert_models/cartpole_0/rollouts/final.pkl"


@pytest.fixture
def venv():
    env_name = "CartPole-v1"
    venv = util.make_vec_env(env_name, 2)
    return venv


@pytest.fixture(params=[32, 50])
def batch_size(request):
    return request.param


@pytest.fixture(params=["data_loader", "ducktyped_data_loader", "transitions"])
def expert_data_type(request):
    return request.param


class DucktypedDataset:
    """Used to check that any iterator over Dict[str, Tensor] works with BC."""

    def __init__(self, transitions: types.TransitionsMinimal, batch_size: int):
        """Builds `DucktypedDataset`."""
        self.trans = transitions
        self.batch_size = batch_size

    def __iter__(self):
        for start in range(0, len(self.trans) - self.batch_size, self.batch_size):
            end = start + self.batch_size
            d = dict(obs=self.trans.obs[start:end], acts=self.trans.acts[start:end])
            d = {k: th.from_numpy(v) for k, v in d.items()}
            yield d


@pytest.fixture
def trainer(batch_size, venv, expert_data_type, custom_logger):
    rollouts = types.load(ROLLOUT_PATH)
    trans = rollout.flatten_trajectories(rollouts)
    if expert_data_type == "data_loader":
        expert_data = th_data.DataLoader(
            trans,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=types.transitions_collate_fn,
        )
    elif expert_data_type == "ducktyped_data_loader":
        expert_data = DucktypedDataset(trans, batch_size)
    elif expert_data_type == "transitions":
        expert_data = trans
    else:  # pragma: no cover
        raise ValueError(expert_data_type)

    return bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        batch_size=batch_size,
        demonstrations=expert_data,
        custom_logger=custom_logger,
    )


def test_weight_decay_init_error(venv, custom_logger):
    with pytest.raises(ValueError, match=".*weight_decay.*"):
        bc.BC(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            demonstrations=None,
            optimizer_kwargs=dict(weight_decay=1e-4),
            custom_logger=custom_logger,
        )


def test_train_end_cond_error(trainer: bc.BC, venv):
    err_context = pytest.raises(ValueError, match="exactly one.*n_epochs")
    with err_context:
        trainer.train(n_epochs=1, n_batches=10)
    with err_context:
        trainer.train()
    with err_context:
        trainer.train(n_epochs=None, n_batches=None)


def test_bc(trainer: bc.BC, venv):
    sample_until = rollout.make_min_episodes(15)
    novice_ret_mean = rollout.mean_return(trainer.policy, venv, sample_until)
    trainer.train(
        n_epochs=1,
        on_epoch_end=lambda: print("epoch end"),
        on_batch_end=lambda: print("batch end"),
    )
    trainer.train(n_batches=10)
    trained_ret_mean = rollout.mean_return(trainer.policy, venv, sample_until)
    # Typically <80 score is bad, >350 is okay. We want an improvement of at
    # least 50 points, which seems like it's not noise.
    assert trained_ret_mean - novice_ret_mean > 50


def test_bc_log_rollouts(trainer: bc.BC, venv):
    trainer.train(n_batches=20, log_rollouts_venv=venv, log_rollouts_n_episodes=1)


class _DataLoaderFailsOnNthIter:
    """A dummy DataLoader that yields after a number of calls of `__iter__`.

    Used by `test_bc_data_loader_empty_iter_error`.
    """

    def __init__(self, dummy_yield_value: dict, no_yield_after_iter: int = 1):
        """Builds dummy data loader.

        Args:
            dummy_yield_value: The value to yield on each call.
            no_yield_after_iter: `__iter__` will raise `StopIteration` after
                this many calls.
        """
        self.iter_count = 0
        self.dummy_yield_value = dummy_yield_value
        self.no_yield_after_iter = no_yield_after_iter

    def __iter__(self):
        if self.iter_count < self.no_yield_after_iter:
            yield self.dummy_yield_value
        self.iter_count += 1


@pytest.mark.parametrize("no_yield_after_iter", [0, 1, 5])
def test_bc_data_loader_empty_iter_error(
    venv: vec_env.VecEnv,
    no_yield_after_iter: bool,
    custom_logger: logger.HierarchicalLogger,
) -> None:
    """Check that we error out if the DataLoader suddenly stops yielding any batches.

    At one point, we entered an updateless infinite loop in this edge case.

    Args:
        venv: Environment to test in.
        no_yield_after_iter: Data loader stops yielding after this many calls.
        custom_logger: Where to log to.
    """
    batch_size = 32
    rollouts = types.load(ROLLOUT_PATH)
    trans = rollout.flatten_trajectories(rollouts)
    dummy_yield_value = dataclasses.asdict(trans[:batch_size])

    bad_data_loader = _DataLoaderFailsOnNthIter(
        dummy_yield_value=dummy_yield_value,
        no_yield_after_iter=no_yield_after_iter,
    )
    trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        batch_size=batch_size,
        custom_logger=custom_logger,
    )
    trainer.set_demonstrations(bad_data_loader)
    with pytest.raises(AssertionError, match=".*no data.*"):
        trainer.train(n_batches=20)


def test_save_reload(trainer, tmpdir):
    pol_path = os.path.join(tmpdir, "policy.pt")
    var_values = list(trainer.policy.parameters())
    trainer.save_policy(pol_path)
    new_policy = bc.reconstruct_policy(pol_path)
    new_values = list(new_policy.parameters())
    assert len(var_values) == len(new_values)
    for old, new in zip(var_values, new_values):
        assert th.allclose(old, new)


def test_train_progress_bar_visibility(trainer: bc.BC):
    """Smoke test for toggling progress bar visibility."""
    for visible in [True, False]:
        trainer.train(n_batches=1, progress_bar=visible)


def test_train_reset_tensorboard(trainer: bc.BC):
    """Smoke test for reset_tensorboard parameter."""
    for reset in [True, False]:
        trainer.train(n_batches=1, reset_tensorboard=reset)
