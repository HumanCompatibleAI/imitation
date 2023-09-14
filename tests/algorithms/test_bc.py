"""Tests `imitation.algorithms.bc`."""

import dataclasses
import os
from typing import Any, Callable, Optional, Sequence

import gym
import hypothesis
import hypothesis.strategies as st
import numpy as np
import pytest
import torch as th
from stable_baselines3.common import envs as sb_envs
from stable_baselines3.common import evaluation
from stable_baselines3.common import policies as sb_policies
from stable_baselines3.common import vec_env

from imitation.algorithms import bc
from imitation.data import rollout, types
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.testing import reward_improvement
from imitation.testing.expert_trajectories import make_expert_transition_loader
from imitation.util import logger, util

########################
# HYPOTHESIS STRATEGIES
########################


def make_bc_train_args(
    on_epoch_end: Callable[[], None],
    on_batch_end: Callable[[], None],
    log_interval: int,
    log_rollouts_n_episodes: int,
    progress_bar: bool,
    reset_tensorboard: bool,
    duration_measure: str,
    duration: int,
    log_rollouts_venv: Optional[vec_env.VecEnv],
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
# Note: we wrap the rngs strategy in a st.shared to ensure that the same RNG is used
# everywhere.
rngs = st.shared(st.builds(np.random.default_rng), key="rng")
env_numbers = st.integers(min_value=1, max_value=10)
envs = st.builds(
    lambda name, num, rng: util.make_vec_env(name, n_envs=num, rng=rng),
    name=env_names,
    num=env_numbers,
    rng=rngs,
)
rollout_envs = st.builds(
    lambda name, num, rng: util.make_vec_env(
        name,
        n_envs=num,
        post_wrappers=[lambda e, _: RolloutInfoWrapper(e)],
        rng=rng,
    ),
    name=env_names,
    num=env_numbers,
    rng=rngs,
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
    lambda env, batch_size, custom_logger, rng: dict(
        observation_space=env.observation_space,
        action_space=env.action_space,
        batch_size=batch_size,
        custom_logger=custom_logger,
        rng=rng,
    ),
    env=envs,
    batch_size=batch_sizes,
    custom_logger=loggers,
    rng=rngs,
)


##############
# SMOKE TESTS
##############


@hypothesis.given(
    env_name=env_names,
    bc_args=bc_args,
    expert_data_type=expert_data_types,
    rng=rngs,
)
# Setting the deadline to none since during the first runs, the expert trajectories must
# be computed. Later they can be loaded from cache much faster.
@hypothesis.settings(deadline=None)
def test_smoke_bc_creation(
    env_name: str,
    bc_args: dict,
    expert_data_type: str,
    rng: np.random.Generator,
    pytestconfig: pytest.Config,
):
    cache = pytestconfig.cache
    assert cache is not None
    bc.BC(
        **bc_args,
        demonstrations=make_expert_transition_loader(
            cache_dir=cache.mkdir("experts"),
            batch_size=bc_args["batch_size"],
            expert_data_type=expert_data_type,
            env_name=env_name,
            rng=rng,
            num_trajectories=60,
        ),
    )


@hypothesis.given(
    env_name=env_names,
    bc_args=bc_args,
    train_args=bc_train_args,
    expert_data_type=expert_data_types,
    rng=rngs,
)
@hypothesis.settings(deadline=20000, max_examples=15)
def test_smoke_bc_training(
    env_name: str,
    bc_args: dict,
    train_args: dict,
    expert_data_type: str,
    rng: np.random.Generator,
    pytestconfig: pytest.Config,
):
    cache = pytestconfig.cache
    assert cache is not None
    # GIVEN
    trainer = bc.BC(
        **bc_args,
        demonstrations=make_expert_transition_loader(
            cache_dir=cache.mkdir("experts"),
            batch_size=bc_args["batch_size"],
            expert_data_type=expert_data_type,
            env_name=env_name,
            rng=rng,
            num_trajectories=2,  # Only use 2 trajectories to speed up the test
        ),
    )
    # WHEN
    trainer.train(**train_args)


#####################
# TEST FUNCTIONALITY
#####################


def test_that_bc_improves_rewards(
    cartpole_bc_trainer: bc.BC,
    cartpole_venv: vec_env.VecEnv,
):
    # GIVEN
    novice_rewards, _ = evaluation.evaluate_policy(
        cartpole_bc_trainer.policy,
        cartpole_venv,
        15,
        return_episode_rewards=True,
    )
    assert isinstance(novice_rewards, list)

    # WHEN
    cartpole_bc_trainer.train(n_epochs=1)
    rewards_after_training, _ = evaluation.evaluate_policy(
        cartpole_bc_trainer.policy,
        cartpole_venv,
        15,
        return_episode_rewards=True,
    )

    # THEN
    assert isinstance(rewards_after_training, list)
    assert reward_improvement.is_significant_reward_improvement(
        novice_rewards,
        rewards_after_training,
    )
    assert reward_improvement.mean_reward_improved_by(
        novice_rewards,
        rewards_after_training,
        50,
    )


def test_gradient_accumulation(
    cartpole_venv: vec_env.VecEnv,
    rng,
    pytestconfig,
):
    batch_size = 6
    minibatch_size = 3
    num_trajectories = 5

    demonstrations = make_expert_transition_loader(
        cache_dir=pytestconfig.cache.makedir("experts"),
        batch_size=6,
        expert_data_type="transitions",
        env_name="seals/CartPole-v0",
        rng=rng,
        num_trajectories=num_trajectories,
    )

    seed = rng.integers(2**32)

    def make_trainer(**kwargs: Any) -> bc.BC:
        th.manual_seed(seed)
        return bc.BC(
            observation_space=cartpole_venv.observation_space,
            action_space=cartpole_venv.action_space,
            batch_size=batch_size,
            demonstrations=demonstrations,
            custom_logger=None,
            rng=rng,
            **kwargs,
        )

    trainers = (make_trainer(), make_trainer(minibatch_size=minibatch_size))

    for step in range(8):
        print("Step", step)
        seed = rng.integers(2**32)

        for trainer in trainers:
            th.manual_seed(seed)
            trainer.train(n_batches=1)

        # Note: due to numerical instability, the models are
        # bound to diverge at some point, but should be stable
        # over the short time frame we test over; however, it is
        # theoretically possible that with very unlucky seeding,
        # this could fail.
        params = zip(trainers[0].policy.parameters(), trainers[1].policy.parameters())
        for p1, p2 in params:
            th.testing.assert_allclose(p1, p2, atol=1e-5, rtol=1e-5)


def test_that_policy_reconstruction_preserves_parameters(
    cartpole_bc_trainer: bc.BC,
    tmpdir,
):
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
        th.testing.assert_close(original, reconstructed)


#############################################
# ENSURE EXCEPTIONS ARE THROWN WHEN EXPECTED
#############################################


def test_that_weight_decay_in_optimizer_raises_error(
    cartpole_venv: vec_env.VecEnv,
    custom_logger: logger.HierarchicalLogger,
    rng: np.random.Generator,
):
    with pytest.raises(ValueError, match=".*weight_decay.*"):
        bc.BC(
            observation_space=cartpole_venv.observation_space,
            action_space=cartpole_venv.action_space,
            demonstrations=None,
            optimizer_kwargs=dict(weight_decay=1e-4),
            custom_logger=custom_logger,
            rng=rng,
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
    cartpole_bc_trainer: bc.BC,
    duration_args: dict,
):
    with pytest.raises(ValueError, match="exactly one.*n_epochs"):
        cartpole_bc_trainer.train(**duration_args)


# Start at 1 as BC uses up an iteration from getting the first element for type checking
@pytest.mark.parametrize("no_yield_after_iter", [1, 2, 6])
def test_that_bc_raises_error_when_data_loader_is_empty(
    no_yield_after_iter: int,
    cartpole_bc_trainer: bc.BC,
    cartpole_expert_trajectories: Sequence[types.TrajectoryWithRew],
    custom_logger: logger.HierarchicalLogger,
) -> None:
    """Check that we error out if the DataLoader suddenly stops yielding any batches.

    At one point, we entered an updateless infinite loop in this edge case.

    Args:
        no_yield_after_iter: Data loader stops yielding after this many calls.
        cartpole_bc_trainer: BC trainer.
        cartpole_expert_trajectories: The expert trajectories to use.
        custom_logger: Where to log to.
    """
    # GIVEN
    batch_size = cartpole_bc_trainer.batch_size
    trans = rollout.flatten_trajectories(cartpole_expert_trajectories)
    dummy_yield_value = dataclasses.asdict(trans[:batch_size])

    class DataLoaderThatFailsOnNthIter:
        """A dummy DataLoader stops to yield after a number of calls to `__iter__`."""

        def __init__(self):
            self.iter_count = 0

        def __iter__(self):
            if self.iter_count < no_yield_after_iter:
                yield dummy_yield_value
            self.iter_count += 1

    batch_cnt = 0

    def inc_batch_cnt():
        nonlocal batch_cnt
        batch_cnt += 1

    # WHEN
    cartpole_bc_trainer.set_demonstrations(DataLoaderThatFailsOnNthIter())
    with pytest.raises(AssertionError, match=".*no data.*"):  # THEN
        cartpole_bc_trainer.train(n_batches=20, on_batch_end=inc_batch_cnt)

    # THEN
    assert batch_cnt == no_yield_after_iter


class FloatReward(gym.RewardWrapper):
    """Typecasts reward to a float."""

    def reward(self, reward):
        return float(reward)


# TODO: make test nicer
def test_dict_space():
    # TODO: is sb_envs okay?
    def make_env():
        env = sb_envs.SimpleMultiObsEnv(channel_last=False)
        return RolloutInfoWrapper(FloatReward(env))

    env = vec_env.DummyVecEnv([make_env, make_env])

    policy = sb_policies.MultiInputActorCriticPolicy(
        env.observation_space,
        env.action_space,
        lambda _: 0.001,
    )
    rng = np.random.default_rng()

    def sample_expert_transitions():
        print("Sampling expert transitions.")
        rollouts = rollout.rollout(
            policy=None,
            venv=env,
            sample_until=rollout.make_sample_until(min_timesteps=None, min_episodes=50),
            rng=rng,
            unwrap=False,  # TODO have rollout unwrap wrapper support dict
        )
        return rollout.flatten_trajectories(rollouts)

    transitions = sample_expert_transitions()

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        policy=policy,
        action_space=env.action_space,
        rng=rng,
        demonstrations=transitions,
    )

    bc_trainer.train(n_epochs=1)

    reward, _ = evaluation.evaluate_policy(
        bc_trainer.policy,  # type: ignore[arg-type]
        env,
        n_eval_episodes=3,
        render=False,  # comment out to speed up
    )
