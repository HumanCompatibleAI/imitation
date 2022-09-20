"""Tests `imitation.algorithms.bc`."""

import dataclasses
import os

import gym
import hypothesis
import hypothesis.strategies as st
import pytest
import torch as th
from conftest import get_expert_trajectories
from stable_baselines3.common import evaluation, vec_env
from torch.utils import data as th_data

from imitation.algorithms import bc
from imitation.data import rollout, types
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.testing import reward_improvement
from imitation.util import logger, util


def make_expert_trajectory_loader(
    pytestconfig,
    batch_size,
    expert_data_type,
    env_name: str,
    max_num_trajectories: int = -1,
):
    trajectories = get_expert_trajectories(pytestconfig, env_name)
    transitions = rollout.flatten_trajectories(
        trajectories[: max(max_num_trajectories, batch_size)],
    )
    if expert_data_type == "data_loader":
        return th_data.DataLoader(
            transitions,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=types.transitions_collate_fn,
        )
    elif expert_data_type == "ducktyped_data_loader":

        class DucktypedDataset:
            """Used to check that any iterator over Dict[str, Tensor] works with BC."""

            def __init__(self, transitions: types.TransitionsMinimal, batch_size: int):
                """Builds `DucktypedDataset`."""
                self.trans = transitions
                self.batch_size = batch_size

            def __iter__(self):
                for start in range(
                    0,
                    len(self.trans) - self.batch_size,
                    self.batch_size,
                ):
                    end = start + self.batch_size
                    d = dict(
                        obs=self.trans.obs[start:end],
                        acts=self.trans.acts[start:end],
                    )
                    d = {k: util.safe_to_tensor(v) for k, v in d.items()}
                    yield d

        return DucktypedDataset(transitions, batch_size)
    elif expert_data_type == "transitions":
        return transitions
    else:  # pragma: no cover
        raise ValueError(expert_data_type)


@pytest.fixture
def cartpole_bc_trainer(
    pytestconfig,
    cartpole_venv,
    cartpole_expert_trajectories,
):
    return bc.BC(
        observation_space=cartpole_venv.observation_space,
        action_space=cartpole_venv.action_space,
        batch_size=50,
        demonstrations=make_expert_trajectory_loader(
            pytestconfig,
            50,
            "transitions",
            "seals/CartPole-v0",
        ),
        custom_logger=None,
    )


# Note: we don't use the Mujoco envs here because mujoco is not installed on CI.
envs = st.sampled_from(["Pendulum-v1", "seals/CartPole-v0"])
batch_sizes = st.integers(min_value=1, max_value=50)
env_numbers = st.integers(min_value=1, max_value=10)
loggers = st.sampled_from([None, logger.configure()])
expert_data_types = st.sampled_from(
    ["data_loader", "ducktyped_data_loader", "transitions"],
)


@st.composite
def bc_train_args(draw):
    args = dict(
        on_epoch_end=draw(st.sampled_from([None, lambda: None])),
        on_batch_end=draw(st.sampled_from([None, lambda: None])),
        log_interval=draw(st.integers(500, 10000)),
        log_rollouts_n_episodes=draw(st.sampled_from([-1, 2])),
        progress_bar=draw(st.booleans()),
        reset_tensorboard=draw(st.booleans()),
    )
    duration_measure = draw(st.sampled_from(["n_batches", "n_epochs"]))
    duration = draw(st.integers(1, 3))
    args[duration_measure] = duration
    return args


@hypothesis.given(
    batch_size=batch_sizes,
    env_name=envs,
    num_envs=env_numbers,
    custom_logger=loggers,
    expert_data_type=expert_data_types,
)
# Setting the deadline to none since during the first runs, the expert trajectories must
# be computed. Later they can be loaded from cache much faster.
@hypothesis.settings(deadline=None)
def test_smoke_bc_creation(
    batch_size,
    env_name,
    num_envs,
    custom_logger,
    expert_data_type,
    pytestconfig,
):
    # GIVEN
    env = vec_env.DummyVecEnv(
        [lambda: RolloutInfoWrapper(gym.make(env_name)) for _ in range(num_envs)],
    )
    bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        batch_size=batch_size,
        demonstrations=make_expert_trajectory_loader(
            pytestconfig,
            batch_size,
            expert_data_type,
            env_name,
        ),
        custom_logger=custom_logger,
    )


@hypothesis.given(
    env_name=envs,
    num_envs=env_numbers,
    train_args=bc_train_args(),
    use_custom_rollout_venv=st.booleans(),
    batch_size=batch_sizes,
    expert_data_type=expert_data_types,
)
@hypothesis.settings(deadline=10000, max_examples=50)
def test_smoke_bc_training(
    env_name,
    num_envs,
    train_args,
    use_custom_rollout_venv,
    batch_size,
    expert_data_type,
    pytestconfig,
):
    # GIVEN
    env = util.make_vec_env(env_name, num_envs)
    trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        batch_size=batch_size,
        demonstrations=make_expert_trajectory_loader(
            pytestconfig,
            batch_size,
            expert_data_type,
            env_name,
            max_num_trajectories=3,  # Only use 3 trajectories to speed up the test
        ),
    )
    if use_custom_rollout_venv:
        custom_rollout_venv = util.make_vec_env(
            env_name,
            num_envs,
            post_wrappers=[lambda e, _: RolloutInfoWrapper(e)],
        )
    else:
        custom_rollout_venv = None

    # WHEN
    trainer.train(log_rollouts_venv=custom_rollout_venv, **train_args)


def test_that_weight_decay_in_optimizer_raises_error(cartpole_venv, custom_logger):
    with pytest.raises(ValueError, match=".*weight_decay.*"):
        bc.BC(
            observation_space=cartpole_venv.observation_space,
            action_space=cartpole_venv.action_space,
            demonstrations=None,
            optimizer_kwargs=dict(weight_decay=1e-4),
            custom_logger=custom_logger,
        )


@pytest.mark.parametrize(
    "duration_args",
    [
        pytest.param(dict(n_epochs=1, n_batches=10), id="both specified"),
        pytest.param(dict(), id="neither specified"),
        pytest.param(dict(n_epochs=None, n_batches=None), id="both None"),
    ],
)
def test_that_wrong_training_duration_specification_raises_error(
    cartpole_bc_trainer,
    duration_args,
):
    with pytest.raises(ValueError, match="exactly one.*n_epochs"):
        cartpole_bc_trainer.train(**duration_args)


def test_that_bc_improves_rewards(cartpole_bc_trainer, cartpole_venv):
    # GIVEN
    novice_rewards, _ = evaluation.evaluate_policy(
        cartpole_bc_trainer.policy,
        cartpole_venv,
        15,
        return_episode_rewards=True,
    )

    # WHEN
    cartpole_bc_trainer.train(n_epochs=1)
    rewards_after_training, _ = evaluation.evaluate_policy(
        cartpole_bc_trainer.policy,
        cartpole_venv,
        15,
        return_episode_rewards=True,
    )

    # THEN
    assert reward_improvement.is_significant_reward_improvement(
        novice_rewards,
        rewards_after_training,
    )
    assert reward_improvement.mean_reward_improved_by(
        novice_rewards,
        rewards_after_training,
        50,
    )


def test_smoke_log_rollouts(cartpole_bc_trainer, cartpole_venv):
    cartpole_bc_trainer.train(
        n_batches=20,
        log_rollouts_venv=cartpole_venv,
        log_rollouts_n_episodes=1,
    )


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
def test_that_bc_raises_error_when_data_loader_is_empty(
    cartpole_venv: vec_env.VecEnv,
    no_yield_after_iter: bool,
    custom_logger: logger.HierarchicalLogger,
    cartpole_expert_trajectories,
) -> None:
    """Check that we error out if the DataLoader suddenly stops yielding any batches.

    At one point, we entered an updateless infinite loop in this edge case.

    Args:
        cartpole_venv: Environment to test in.
        no_yield_after_iter: Data loader stops yielding after this many calls.
        custom_logger: Where to log to.
        cartpole_expert_trajectories: The expert trajectories to use.
    """
    batch_size = 32
    trans = rollout.flatten_trajectories(cartpole_expert_trajectories)
    dummy_yield_value = dataclasses.asdict(trans[:batch_size])

    bad_data_loader = _DataLoaderFailsOnNthIter(
        dummy_yield_value=dummy_yield_value,
        no_yield_after_iter=no_yield_after_iter,
    )
    trainer = bc.BC(
        observation_space=cartpole_venv.observation_space,
        action_space=cartpole_venv.action_space,
        batch_size=batch_size,
        custom_logger=custom_logger,
    )
    trainer.set_demonstrations(bad_data_loader)
    with pytest.raises(AssertionError, match=".*no data.*"):
        trainer.train(n_batches=20)


def test_save_reload(cartpole_bc_trainer, tmpdir):
    pol_path = os.path.join(tmpdir, "policy.pt")
    var_values = list(cartpole_bc_trainer.policy.parameters())
    cartpole_bc_trainer.save_policy(pol_path)
    new_policy = bc.reconstruct_policy(pol_path)
    new_values = list(new_policy.parameters())
    assert len(var_values) == len(new_values)
    for old, new in zip(var_values, new_values):
        assert th.allclose(old, new)


def test_train_progress_bar_visibility(cartpole_bc_trainer):
    """Smoke test for toggling progress bar visibility."""
    for visible in [True, False]:
        cartpole_bc_trainer.train(n_batches=1, progress_bar=visible)


def test_train_reset_tensorboard(cartpole_bc_trainer):
    """Smoke test for reset_tensorboard parameter."""
    for reset in [True, False]:
        cartpole_bc_trainer.train(n_batches=1, reset_tensorboard=reset)
