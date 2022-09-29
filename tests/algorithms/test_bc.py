"""Tests `imitation.algorithms.bc`."""

import dataclasses
import os

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

#########
# UTILS
#########


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


########################
# HYPOTHESIS STRATEGIES
########################


def make_bc_train_args(
    on_epoch_end,
    on_batch_end,
    log_interval,
    log_rollouts_n_episodes,
    progress_bar,
    reset_tensorboard,
    duration_measure,
    duration,
    log_rollouts_venv,
):
    return {
        "on_epoch_end": on_epoch_end,
        "on_batch_end": on_batch_end,
        "log_interval": log_interval,
        "log_rollouts_n_episodes": log_rollouts_n_episodes,
        "progress_bar": progress_bar,
        "reset_tensorboard": reset_tensorboard,
        duration_measure: duration,
        "log_rollouts_venv": log_rollouts_venv,
    }


# Note: we don't use the Mujoco envs here because mujoco is not installed on CI.
# Note: we wrap the env_names strategy in a st.shared to ensure that the same env name
# is chosen for BC creation, expert data loading, and policy evaluation.
env_names = st.shared(
    st.sampled_from(["Pendulum-v1", "seals/CartPole-v0"]),
    key="env_name",
)
env_numbers = st.integers(min_value=1, max_value=10)
envs = st.builds(
    lambda name, num: util.make_vec_env(name, num),
    name=env_names,
    num=env_numbers,
)
rollout_envs = st.builds(
    lambda name, num: util.make_vec_env(
        name,
        num,
        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)],
    ),
    name=env_names,
    num=env_numbers,
)
batch_sizes = st.integers(min_value=1, max_value=50)
loggers = st.sampled_from([None, logger.configure()])
expert_data_types = st.sampled_from(
    ["data_loader", "ducktyped_data_loader", "transitions"],
)
bc_train_args = st.builds(
    make_bc_train_args,
    on_epoch_end=st.sampled_from([None, lambda: None]),
    on_batch_end=st.sampled_from([None, lambda: None]),
    log_interval=st.integers(500, 10000),
    log_rollouts_n_episodes=st.sampled_from([-1, 1, 2]),
    progress_bar=st.booleans(),
    reset_tensorboard=st.booleans(),
    duration_measure=st.sampled_from(["n_batches", "n_epochs"]),
    duration=st.integers(1, 3),
    log_rollouts_venv=st.one_of(rollout_envs, st.just(None)),
)
bc_args = st.builds(
    lambda env, batch_size, custom_logger: dict(
        observation_space=env.observation_space,
        action_space=env.action_space,
        batch_size=batch_size,
        custom_logger=custom_logger,
    ),
    env=envs,
    batch_size=batch_sizes,
    custom_logger=loggers,
)


##############
# SMOKE TESTS
##############


@hypothesis.given(
    env_name=env_names,
    bc_args=bc_args,
    expert_data_type=expert_data_types,
)
# Setting the deadline to none since during the first runs, the expert trajectories must
# be computed. Later they can be loaded from cache much faster.
@hypothesis.settings(deadline=None)
def test_smoke_bc_creation(
    env_name,
    bc_args,
    expert_data_type,
    pytestconfig,
):
    bc.BC(
        **bc_args,
        demonstrations=make_expert_trajectory_loader(
            pytestconfig,
            bc_args["batch_size"],
            expert_data_type,
            env_name,
        ),
    )


@hypothesis.given(
    env_name=env_names,
    bc_args=bc_args,
    train_args=bc_train_args,
    expert_data_type=expert_data_types,
)
@hypothesis.settings(deadline=10000, max_examples=50)
def test_smoke_bc_training(
    env_name,
    bc_args,
    train_args,
    expert_data_type,
    pytestconfig,
):
    # GIVEN
    trainer = bc.BC(
        **bc_args,
        demonstrations=make_expert_trajectory_loader(
            pytestconfig,
            bc_args["batch_size"],
            expert_data_type,
            env_name,
            max_num_trajectories=3,  # Only use 3 trajectories to speed up the test
        ),
    )
    # WHEN
    trainer.train(**train_args)


#####################
# TEST FUNCTIONALITY
#####################


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


def test_that_policy_reconstruction_preserves_parameters(cartpole_bc_trainer, tmpdir):
    # GIVEN
    pol_path = os.path.join(tmpdir, "policy.pt")
    original_parameters = list(cartpole_bc_trainer.policy.parameters())

    # WHEN
    cartpole_bc_trainer.save_policy(pol_path)
    reconstructed_policy = bc.reconstruct_policy(pol_path)

    # THEN
    reconstructed_parameters = list(reconstructed_policy.parameters())
    assert len(original_parameters) == len(reconstructed_parameters)
    for original, reconstructed in zip(original_parameters, reconstructed_parameters):
        assert th.allclose(original, reconstructed)


#############################################
# ENSURE EXCEPTIONS ARE THROWN WHEN EXPECTED
#############################################


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


class _DataLoaderThatFailsOnNthIter:
    """A dummy DataLoader stops to yield after a number of calls to `__iter__`."""

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

    bad_data_loader = _DataLoaderThatFailsOnNthIter(
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
