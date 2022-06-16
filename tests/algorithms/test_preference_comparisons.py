"""Tests for the preference comparisons reward learning implementation."""

import re
from typing import Sequence

import numpy as np
import pytest
import seals  # noqa: F401
import stable_baselines3

from imitation.algorithms import preference_comparisons
from imitation.data import types
from imitation.data.types import TrajectoryWithRew
from imitation.rewards import reward_nets
from imitation.util import util


@pytest.fixture
def venv():
    return util.make_vec_env(
        "seals/CartPole-v0",
        n_envs=1,
    )


@pytest.fixture
def reward_net(venv):
    return reward_nets.BasicRewardNet(venv.observation_space, venv.action_space)


@pytest.fixture
def agent(venv):
    return stable_baselines3.PPO(
        "MlpPolicy",
        venv,
        n_epochs=1,
        batch_size=2,
        n_steps=10,
    )


@pytest.fixture
def fragmenter():
    return preference_comparisons.RandomFragmenter(seed=0, warning_threshold=0)


@pytest.fixture
def agent_trainer(agent, reward_net):
    return preference_comparisons.AgentTrainer(agent, reward_net)


def _check_trajs_equal(
    trajs1: Sequence[types.TrajectoryWithRew],
    trajs2: Sequence[types.TrajectoryWithRew],
):
    assert len(trajs1) == len(trajs2)
    for traj1, traj2 in zip(trajs1, trajs2):
        assert np.array_equal(traj1.obs, traj2.obs)
        assert np.array_equal(traj1.acts, traj2.acts)
        assert np.array_equal(traj1.rews, traj2.rews)
        assert np.array_equal(traj1.infos, traj2.infos)
        assert traj1.terminal == traj2.terminal


def test_missing_environment(agent):
    # Create an agent that doesn't have its environment set.
    # More realistically, this can happen when loading a stored agent.
    agent.env = None
    with pytest.raises(
        ValueError,
        match="The environment for the agent algorithm must be set.",
    ):
        preference_comparisons.AgentTrainer(agent, reward_net)


def test_trajectory_dataset_seeding(
    cartpole_expert_trajectories: Sequence[TrajectoryWithRew],
    num_samples: int = 400,
):
    dataset1 = preference_comparisons.TrajectoryDataset(
        cartpole_expert_trajectories,
        seed=0,
    )
    sample1 = dataset1.sample(num_samples)
    dataset2 = preference_comparisons.TrajectoryDataset(
        cartpole_expert_trajectories,
        seed=0,
    )
    sample2 = dataset2.sample(num_samples)

    _check_trajs_equal(sample1, sample2)

    dataset3 = preference_comparisons.TrajectoryDataset(
        cartpole_expert_trajectories,
        seed=42,
    )
    sample3 = dataset3.sample(num_samples)
    with pytest.raises(AssertionError):
        _check_trajs_equal(sample2, sample3)


# CartPole max episode length is 200
@pytest.mark.parametrize("num_steps", [0, 199, 200, 201, 400])
def test_trajectory_dataset_len(
    cartpole_expert_trajectories: Sequence[TrajectoryWithRew],
    num_steps: int,
):
    dataset = preference_comparisons.TrajectoryDataset(
        cartpole_expert_trajectories,
        seed=0,
    )
    sample = dataset.sample(num_steps)
    lengths = [len(t) for t in sample]
    assert sum(lengths) >= num_steps
    if num_steps > 0:
        assert sum(lengths) - min(lengths) < num_steps


def test_trajectory_dataset_too_long(
    cartpole_expert_trajectories: Sequence[TrajectoryWithRew],
):
    dataset = preference_comparisons.TrajectoryDataset(
        cartpole_expert_trajectories,
        seed=0,
    )
    with pytest.raises(RuntimeError, match="Asked for.*but only.* available"):
        dataset.sample(100000)


def test_trajectory_dataset_shuffle(
    cartpole_expert_trajectories: Sequence[TrajectoryWithRew],
    num_steps: int = 400,
):
    dataset = preference_comparisons.TrajectoryDataset(
        cartpole_expert_trajectories,
        seed=0,
    )
    sample = dataset.sample(num_steps)
    sample2 = dataset.sample(num_steps)
    with pytest.raises(AssertionError):
        _check_trajs_equal(sample, sample2)


def test_transitions_left_in_buffer(agent_trainer):
    # Faster to just set the counter than to actually fill the buffer
    # with transitions.
    agent_trainer.buffering_wrapper.n_transitions = 2
    with pytest.raises(
        RuntimeError,
        match=re.escape(
            "There are 2 transitions left in the buffer. "
            "Call AgentTrainer.sample() first to clear them.",
        ),
    ):
        agent_trainer.train(steps=1)


def test_trainer_no_crash(agent_trainer, reward_net, fragmenter, custom_logger):
    main_trainer = preference_comparisons.PreferenceComparisons(
        agent_trainer,
        reward_net,
        transition_oversampling=2,
        fragment_length=2,
        comparisons_per_iteration=2,
        fragmenter=fragmenter,
        custom_logger=custom_logger,
    )
    result = main_trainer.train(10, 3)
    # We don't expect good performance after training for 10 (!) timesteps,
    # but check stats are within the bounds they should lie in.
    assert result["reward_loss"] > 0.0
    assert 0.0 < result["reward_accuracy"] < 1.0


def test_discount_rate_no_crash(agent_trainer, reward_net, fragmenter, custom_logger):
    # also use a non-zero noise probability to check that doesn't cause errors
    reward_trainer = preference_comparisons.CrossEntropyRewardTrainer(
        reward_net,
        noise_prob=0.1,
        discount_factor=0.9,
    )
    main_trainer = preference_comparisons.PreferenceComparisons(
        agent_trainer,
        reward_net,
        transition_oversampling=2,
        fragment_length=2,
        comparisons_per_iteration=2,
        fragmenter=fragmenter,
        reward_trainer=reward_trainer,
        custom_logger=custom_logger,
    )
    main_trainer.train(10, 3)


def test_synthetic_gatherer_deterministic(agent_trainer, fragmenter):
    gatherer = preference_comparisons.SyntheticGatherer(temperature=0)
    trajectories = agent_trainer.sample(10)
    fragments = fragmenter(trajectories, fragment_length=2, num_pairs=2)
    preferences1 = gatherer(fragments)
    preferences2 = gatherer(fragments)
    assert np.all(preferences1 == preferences2)


def test_fragments_terminal(fragmenter):
    trajectories = [
        types.TrajectoryWithRew(
            obs=np.arange(4),
            acts=np.zeros((3,)),
            rews=np.zeros((3,)),
            infos=None,
            terminal=True,
        ),
        types.TrajectoryWithRew(
            obs=np.arange(3),
            acts=np.zeros((2,)),
            rews=np.zeros((2,)),
            infos=None,
            terminal=False,
        ),
    ]
    for _ in range(5):
        for frags in fragmenter(trajectories, fragment_length=2, num_pairs=2):
            for frag in frags:
                assert (frag.obs[-1] == 3) == frag.terminal


def test_fragments_too_short_error(agent_trainer):
    trajectories = agent_trainer.sample(2)
    fragmenter = preference_comparisons.RandomFragmenter(
        seed=0,
        warning_threshold=0,
    )
    with pytest.raises(
        ValueError,
        match="No trajectories are long enough for the desired fragment length.",
    ):
        # the only important bit is that fragment_length is higher than
        # we'll ever reach
        fragmenter(trajectories, fragment_length=10000, num_pairs=2)


def test_preference_dataset_errors(agent_trainer, fragmenter):
    dataset = preference_comparisons.PreferenceDataset()
    trajectories = agent_trainer.sample(2)
    fragments = fragmenter(trajectories, fragment_length=2, num_pairs=2)
    # just create something with a different shape:
    preferences = np.empty(len(fragments) + 1, dtype=np.float32)
    with pytest.raises(ValueError, match="Unexpected preferences shape"):
        dataset.push(fragments, preferences)

    # Now test dtype
    preferences = np.empty(len(fragments), dtype=np.float64)
    with pytest.raises(ValueError, match="preferences should have dtype float32"):
        dataset.push(fragments, preferences)


def test_store_and_load_preference_dataset(agent_trainer, fragmenter, tmp_path):
    dataset = preference_comparisons.PreferenceDataset()
    trajectories = agent_trainer.sample(10)
    fragments = fragmenter(trajectories, fragment_length=2, num_pairs=2)
    gatherer = preference_comparisons.SyntheticGatherer()
    preferences = gatherer(fragments)
    dataset.push(fragments, preferences)

    path = tmp_path / "preferences.pkl"
    dataset.save(path)
    loaded = preference_comparisons.PreferenceDataset.load(path)
    assert len(loaded) == len(dataset)
    for sample, loaded_sample in zip(dataset, loaded):
        fragments, preference = sample
        loaded_fragments, loaded_preference = loaded_sample

        assert preference == loaded_preference
        _check_trajs_equal(fragments, loaded_fragments)


def test_exploration_no_crash(agent, reward_net, fragmenter, custom_logger):
    agent_trainer = preference_comparisons.AgentTrainer(
        agent,
        reward_net,
        exploration_frac=0.5,
    )
    main_trainer = preference_comparisons.PreferenceComparisons(
        agent_trainer,
        reward_net,
        transition_oversampling=2,
        fragment_length=5,
        comparisons_per_iteration=2,
        fragmenter=fragmenter,
        custom_logger=custom_logger,
    )
    main_trainer.train(10, 3)
