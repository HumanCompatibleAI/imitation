"""Tests for the preference comparisons reward learning implementation."""

import math
import re
import uuid
from typing import Any, Sequence, Tuple
from unittest.mock import Mock, MagicMock

import gym
import numpy as np
import pytest
import seals  # noqa: F401
import stable_baselines3
import torch as th
from gym import spaces
from stable_baselines3.common import evaluation
from stable_baselines3.common.envs import FakeImageEnv
from stable_baselines3.common.vec_env import DummyVecEnv

import imitation.testing.reward_nets as testing_reward_nets
from imitation.algorithms import preference_comparisons
from imitation.algorithms.preference_comparisons import PreferenceQuerent, PrefCollectQuerent, PreferenceGatherer, \
    SyntheticGatherer, PrefCollectGatherer
from imitation.data import types
from imitation.data.types import TrajectoryWithRew
from imitation.regularization import regularizers, updaters
from imitation.rewards import reward_nets
from imitation.util import networks, util

UNCERTAINTY_ON = ["logit", "probability", "label"]


@pytest.fixture
def venv(rng):
    return util.make_vec_env(
        "seals/CartPole-v0",
        n_envs=1,
        rng=rng,
    )


@pytest.fixture(
    params=[
        reward_nets.BasicRewardNet,
        testing_reward_nets.make_ensemble,
        lambda *args: reward_nets.AddSTDRewardWrapper(
            testing_reward_nets.make_ensemble(*args),
        ),
    ],
)
def reward_net(request, venv):
    return request.param(venv.observation_space, venv.action_space)


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
def random_fragmenter(rng):
    return preference_comparisons.RandomFragmenter(
        rng=rng,
        warning_threshold=0,
    )


@pytest.fixture
def agent_trainer(agent, reward_net, venv, rng):
    return preference_comparisons.AgentTrainer(agent, reward_net, venv, rng)


# TODO: trajectory_with_rew fixture already exists in data.test_types, should be moved to a conftest.py
@pytest.fixture
def trajectory_with_rew(venv):
    observations, rewards, dones, infos, actions = [], [], [], [], []
    observations.append(venv.observation_space.sample())
    for _ in range(2):
        observations.append(venv.observation_space.sample())
        actions.append(venv.action_space.sample())
        rewards.append(0.0)
        infos.append({})
    return TrajectoryWithRew(obs=np.array(observations),
                             acts=np.array(actions),
                             rews=np.array(rewards),
                             infos=np.array(infos),
                             terminal=False)


def assert_info_arrs_equal(arr1, arr2):  # pragma: no cover
    def check_possibly_nested_dicts_equal(dict1, dict2):
        for key, val1 in dict1.items():
            val2 = dict2[key]
            if isinstance(val1, dict):
                check_possibly_nested_dicts_equal(val1, val2)
            else:
                assert np.array_equal(val1, val2)

    for item1, item2 in zip(arr1, arr2):
        assert isinstance(item1, dict)
        assert isinstance(item2, dict)
        check_possibly_nested_dicts_equal(item1, item2)


def _check_trajs_equal(
        trajs1: Sequence[types.TrajectoryWithRew],
        trajs2: Sequence[types.TrajectoryWithRew],
):
    assert len(trajs1) == len(trajs2)
    for traj1, traj2 in zip(trajs1, trajs2):
        assert np.array_equal(traj1.obs, traj2.obs)
        assert np.array_equal(traj1.acts, traj2.acts)
        assert np.array_equal(traj1.rews, traj2.rews)
        assert traj1.infos is not None
        assert traj2.infos is not None
        assert_info_arrs_equal(traj1.infos, traj2.infos)
        assert traj1.terminal == traj2.terminal


def test_mismatched_spaces(venv, agent, rng):
    other_venv = util.make_vec_env(
        "seals/MountainCar-v0",
        n_envs=1,
        rng=rng,
    )
    bad_reward_net = reward_nets.BasicRewardNet(
        other_venv.observation_space,
        other_venv.action_space,
    )
    with pytest.raises(
            ValueError,
            match="spaces do not match",
    ):
        preference_comparisons.AgentTrainer(
            agent,
            bad_reward_net,
            venv,
            rng=rng,
        )


def test_trajectory_dataset_seeding(
        cartpole_expert_trajectories: Sequence[TrajectoryWithRew],
        num_samples: int = 400,
):
    dataset1 = preference_comparisons.TrajectoryDataset(
        cartpole_expert_trajectories,
        rng=np.random.default_rng(0),
    )
    sample1 = dataset1.sample(num_samples)
    dataset2 = preference_comparisons.TrajectoryDataset(
        cartpole_expert_trajectories,
        rng=np.random.default_rng(0),
    )
    sample2 = dataset2.sample(num_samples)

    _check_trajs_equal(sample1, sample2)

    dataset3 = preference_comparisons.TrajectoryDataset(
        cartpole_expert_trajectories,
        rng=np.random.default_rng(42),
    )
    sample3 = dataset3.sample(num_samples)
    with pytest.raises(AssertionError):
        _check_trajs_equal(sample2, sample3)


# CartPole max episode length is 200
@pytest.mark.parametrize("num_steps", [0, 199, 200, 201, 400])
def test_trajectory_dataset_len(
        cartpole_expert_trajectories: Sequence[TrajectoryWithRew],
        num_steps: int,
        rng,
):
    dataset = preference_comparisons.TrajectoryDataset(
        cartpole_expert_trajectories,
        rng=rng,
    )
    sample = dataset.sample(num_steps)
    lengths = [len(t) for t in sample]
    assert sum(lengths) >= num_steps
    if num_steps > 0:
        assert sum(lengths) - min(lengths) < num_steps


def test_trajectory_dataset_too_long(
        cartpole_expert_trajectories: Sequence[TrajectoryWithRew],
        rng,
):
    dataset = preference_comparisons.TrajectoryDataset(
        cartpole_expert_trajectories,
        rng=rng,
    )
    with pytest.raises(RuntimeError, match="Asked for.*but only.* available"):
        dataset.sample(100000)


def test_trajectory_dataset_not_static(
        cartpole_expert_trajectories: Sequence[TrajectoryWithRew],
        rng,
        num_steps: int = 400,
):
    """Tests sample() doesn't always return the same value."""
    dataset = preference_comparisons.TrajectoryDataset(
        cartpole_expert_trajectories,
        rng,
    )
    n_trajectories = len(cartpole_expert_trajectories)
    flakiness_prob = 1 / n_trajectories
    max_flakiness = 1e-6
    # Choose max_samples s.t. flakiness_prob**max_samples <= max_flakiness
    max_samples = math.ceil(math.log(max_flakiness) / math.log(flakiness_prob))
    sample = dataset.sample(num_steps)
    with pytest.raises(AssertionError):
        for _ in range(max_samples):
            sample2 = dataset.sample(num_steps)
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


@pytest.mark.parametrize(
    "schedule",
    ["constant", "hyperbolic", "inverse_quadratic", lambda t: 1 / (1 + t ** 3)],
)
def test_preference_comparisons_raises(
        agent_trainer,
        reward_net,
        random_fragmenter,
        preference_model,
        custom_logger,
        schedule,
        rng,
):
    loss = preference_comparisons.CrossEntropyRewardLoss()
    reward_trainer = preference_comparisons.BasicRewardTrainer(
        preference_model,
        loss,
        rng=rng,
    )
    gatherer = preference_comparisons.SyntheticGatherer(rng=rng)
    # no rng, must provide fragmenter, preference gatherer, reward trainer
    no_rng_msg = (
        ".*don't provide.*random state.*provide.*fragmenter"
        ".*preference gatherer.*reward_trainer.*"
    )

    def build_preference_comparsions(gatherer, reward_trainer, fragmenter, rng):
        preference_comparisons.PreferenceComparisons(
            agent_trainer,
            reward_net,
            num_iterations=2,
            transition_oversampling=2,
            reward_trainer=reward_trainer,
            preference_gatherer=gatherer,
            fragmenter=fragmenter,
            custom_logger=custom_logger,
            query_schedule=schedule,
            rng=rng,
        )

    with pytest.raises(ValueError, match=no_rng_msg):
        build_preference_comparsions(gatherer, None, None, rng=None)

    with pytest.raises(ValueError, match=no_rng_msg):
        build_preference_comparsions(None, reward_trainer, None, rng=None)

    with pytest.raises(ValueError, match=no_rng_msg):
        build_preference_comparsions(None, None, random_fragmenter, rng=None)

    # This should not raise
    build_preference_comparsions(gatherer, reward_trainer, random_fragmenter, rng=None)

    # if providing fragmenter, preference gatherer, reward trainer, does not need rng.
    with_rng_msg = (
        "provide.*fragmenter.*preference gatherer.*reward trainer"
        ".*don't need.*random state.*"
    )

    with pytest.raises(ValueError, match=with_rng_msg):
        build_preference_comparsions(
            gatherer,
            reward_trainer,
            random_fragmenter,
            rng=rng,
        )

    # This should not raise
    build_preference_comparsions(None, None, None, rng=rng)
    build_preference_comparsions(gatherer, None, None, rng=rng)
    build_preference_comparsions(None, reward_trainer, None, rng=rng)
    build_preference_comparsions(None, None, random_fragmenter, rng=rng)


@pytest.mark.parametrize(
    "schedule",
    ["constant", "hyperbolic", "inverse_quadratic", lambda t: 1 / (1 + t ** 3)],
)
def test_trainer_no_crash(
        agent_trainer,
        reward_net,
        random_fragmenter,
        custom_logger,
        schedule,
        rng,
):
    main_trainer = preference_comparisons.PreferenceComparisons(
        agent_trainer,
        reward_net,
        num_iterations=2,
        transition_oversampling=2,
        fragment_length=2,
        fragmenter=random_fragmenter,
        custom_logger=custom_logger,
        query_schedule=schedule,
        initial_epoch_multiplier=2,
        rng=rng,
    )
    result = main_trainer.train(100, 10)
    # We don't expect good performance after training for 10 (!) timesteps,
    # but check stats are within the bounds they should lie in.
    assert result["reward_loss"] > 0.0
    assert 0.0 < result["reward_accuracy"] <= 1.0


def test_reward_ensemble_trainer_raises_type_error(venv, rng):
    reward_net = reward_nets.BasicRewardNet(venv.observation_space, venv.action_space)
    preference_model = preference_comparisons.PreferenceModel(
        model=reward_net,
        noise_prob=0.1,
        discount_factor=0.9,
        threshold=50,
    )
    loss = preference_comparisons.CrossEntropyRewardLoss()

    with pytest.raises(
            TypeError,
            match=r"PreferenceModel of a RewardEnsemble expected by EnsembleTrainer.",
    ):
        preference_comparisons.EnsembleTrainer(
            preference_model,
            loss,
            rng=rng,
        )


def test_correct_reward_trainer_used_by_default(
        agent_trainer,
        reward_net,
        random_fragmenter,
        custom_logger,
        rng,
):
    main_trainer = preference_comparisons.PreferenceComparisons(
        agent_trainer,
        reward_net,
        num_iterations=2,
        transition_oversampling=2,
        fragment_length=2,
        fragmenter=random_fragmenter,
        rng=rng,
        custom_logger=custom_logger,
    )

    base_reward_net = reward_net.base if hasattr(reward_net, "base") else reward_net
    if isinstance(base_reward_net, reward_nets.RewardEnsemble):
        assert isinstance(
            main_trainer.reward_trainer,
            preference_comparisons.EnsembleTrainer,
        )
    else:
        assert isinstance(
            main_trainer.reward_trainer,
            preference_comparisons.BasicRewardTrainer,
        )


def test_init_raises_error_when_trying_use_improperly_wrapped_ensemble(
        agent_trainer,
        venv,
        random_fragmenter,
        custom_logger,
        rng,
):
    reward_net = testing_reward_nets.make_ensemble(
        venv.observation_space,
        venv.action_space,
    )
    reward_net = reward_nets.NormalizedRewardNet(reward_net, networks.RunningNorm)
    rgx = (
        r"RewardEnsemble can only be wrapped by "
        r"AddSTDRewardWrapper but found NormalizedRewardNet."
    )
    with pytest.raises(
            ValueError,
            match=rgx,
    ):
        preference_comparisons.PreferenceComparisons(
            agent_trainer,
            reward_net,
            num_iterations=2,
            transition_oversampling=2,
            fragment_length=2,
            fragmenter=random_fragmenter,
            rng=rng,
            custom_logger=custom_logger,
        )


def test_discount_rate_no_crash(
        agent_trainer,
        venv,
        random_fragmenter,
        custom_logger,
        rng,
):
    # also use a non-zero noise probability to check that doesn't cause errors
    reward_net = reward_nets.BasicRewardNet(venv.observation_space, venv.action_space)
    preference_model = preference_comparisons.PreferenceModel(
        model=reward_net,
        noise_prob=0.1,
        discount_factor=0.9,
        threshold=50,
    )
    loss = preference_comparisons.CrossEntropyRewardLoss()
    reward_trainer = preference_comparisons.BasicRewardTrainer(
        preference_model,
        loss,
        rng=rng,
    )

    main_trainer = preference_comparisons.PreferenceComparisons(
        agent_trainer,
        reward_net,
        num_iterations=2,
        transition_oversampling=2,
        fragment_length=2,
        fragmenter=random_fragmenter,
        rng=rng,
        reward_trainer=reward_trainer,
        custom_logger=custom_logger,
    )
    main_trainer.train(100, 10)


def create_reward_trainer(
        venv,
        seed: int,
        batch_size: int,
        **kwargs: Any,
):
    th.manual_seed(seed)
    reward_net = reward_nets.BasicRewardNet(venv.observation_space, venv.action_space)
    preference_model = preference_comparisons.PreferenceModel(model=reward_net)
    loss = preference_comparisons.CrossEntropyRewardLoss()
    rng = np.random.default_rng(seed)
    reward_trainer = preference_comparisons.BasicRewardTrainer(
        preference_model,
        loss,
        rng=rng,
        batch_size=batch_size,
        **kwargs,
    )
    return reward_trainer, reward_net


def test_gradient_accumulation(
        agent_trainer,
        venv,
        random_fragmenter,
        custom_logger,
        rng,
):
    # Test that training steps on the same dataset with different minibatch sizes
    # result in the same reward network.
    batch_size = 6
    minibatch_size = 3
    num_trajectories = 5

    preference_gatherer = preference_comparisons.SyntheticGatherer(
        custom_logger=custom_logger,
        rng=rng,
    )
    dataset = preference_comparisons.PreferenceDataset()
    trajectory = agent_trainer.sample(num_trajectories)
    fragments = random_fragmenter(trajectory, 1, num_trajectories)
    preferences = preference_gatherer(fragments)
    dataset.push(fragments, preferences)

    seed = rng.integers(2 ** 32)
    reward_trainer1, reward_net1 = create_reward_trainer(venv, seed, batch_size)
    reward_trainer2, reward_net2 = create_reward_trainer(
        venv,
        seed,
        batch_size,
        minibatch_size=minibatch_size,
    )

    for step in range(8):
        print("Step", step)
        seed = rng.integers(2 ** 32)

        th.manual_seed(seed)
        reward_trainer1.train(dataset)

        th.manual_seed(seed)
        reward_trainer2.train(dataset)

        # Note: due to numerical instability, the models are
        # bound to diverge at some point, but should be stable
        # over the short time frame we test over; however, it is
        # theoretically possible that with very unlucky seeding,
        # this could fail.
        atol = 1e-5
        rtol = 1e-4
        for p1, p2 in zip(reward_net1.parameters(), reward_net2.parameters()):
            th.testing.assert_close(p1, p2, atol=atol, rtol=rtol)


def test_synthetic_gatherer_deterministic(
        agent_trainer,
        random_fragmenter,
        rng,
):
    gatherer = preference_comparisons.SyntheticGatherer(
        temperature=0,
        rng=rng,
    )
    trajectories = agent_trainer.sample(10)
    fragments = random_fragmenter(trajectories, fragment_length=2, num_pairs=2)
    preferences1 = gatherer(fragments)
    preferences2 = gatherer(fragments)
    assert np.all(preferences1 == preferences2)


def test_synthetic_gatherer_raises(
        agent_trainer,
        random_fragmenter,
):
    with pytest.raises(
            ValueError,
            match="If `sample` is True, then `rng` must be provided",
    ):
        preference_comparisons.SyntheticGatherer(
            temperature=0,
            sample=True,
        )


def test_fragments_terminal(random_fragmenter):
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
        for frags in random_fragmenter(trajectories, fragment_length=2, num_pairs=2):
            for frag in frags:
                assert (frag.obs[-1] == 3) == frag.terminal


def test_fragments_too_short_error(agent_trainer):
    trajectories = agent_trainer.sample(2)
    random_fragmenter = preference_comparisons.RandomFragmenter(
        rng=np.random.default_rng(0),
        warning_threshold=0,
    )
    with pytest.raises(
            ValueError,
            match="No trajectories are long enough for the desired fragment length.",
    ):
        # the only important bit is that fragment_length is higher than
        # we'll ever reach
        random_fragmenter(trajectories, fragment_length=10000, num_pairs=2)


def test_preference_dataset_errors(agent_trainer, random_fragmenter):
    dataset = preference_comparisons.PreferenceDataset()
    trajectories = agent_trainer.sample(2)
    fragments = random_fragmenter(trajectories, fragment_length=2, num_pairs=2)
    # just create something with a different shape:
    preferences = np.empty(len(fragments) + 1, dtype=np.float32)
    with pytest.raises(ValueError, match="Unexpected preferences shape"):
        dataset.push(fragments, preferences)

    # Now test dtype
    preferences = np.empty(len(fragments), dtype=np.float64)
    with pytest.raises(ValueError, match="preferences should have dtype float32"):
        dataset.push(fragments, preferences)


# TODO: update test
def test_preference_dataset_queue(agent_trainer, random_fragmenter, rng):
    dataset = preference_comparisons.PreferenceDataset(max_size=5)
    trajectories = agent_trainer.sample(10)

    gatherer = preference_comparisons.SyntheticGatherer(rng=rng)
    for i in range(6):
        fragments = random_fragmenter(trajectories, fragment_length=2, num_pairs=1)
        preferences = gatherer(fragments)
        assert len(dataset) == min(i, 5)
        dataset.push(fragments, preferences)
        assert len(dataset) == min(i + 1, 5)

    # The first comparison should have been evicted to keep the size at 5
    assert len(dataset) == 5


def test_store_and_load_preference_dataset(
        agent_trainer,
        random_fragmenter,
        tmp_path,
        rng,
):
    dataset = preference_comparisons.PreferenceDataset()
    trajectories = agent_trainer.sample(10)
    fragments = random_fragmenter(trajectories, fragment_length=2, num_pairs=2)
    gatherer = preference_comparisons.SyntheticGatherer(rng=rng)
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


def test_exploration_no_crash(
        agent,
        reward_net,
        venv,
        random_fragmenter,
        custom_logger,
        rng,
):
    agent_trainer = preference_comparisons.AgentTrainer(
        agent,
        reward_net,
        venv,
        rng=rng,
        exploration_frac=0.5,
    )
    main_trainer = preference_comparisons.PreferenceComparisons(
        agent_trainer,
        reward_net,
        num_iterations=2,
        transition_oversampling=2,
        fragment_length=5,
        fragmenter=random_fragmenter,
        rng=rng,
        custom_logger=custom_logger,
    )
    main_trainer.train(100, 10)


@pytest.mark.parametrize("uncertainty_on", UNCERTAINTY_ON)
def test_active_fragmenter_discount_rate_no_crash(
        agent_trainer,
        venv,
        random_fragmenter,
        uncertainty_on,
        custom_logger,
        rng,
):
    # also use a non-zero noise probability to check that doesn't cause errors
    reward_net = reward_nets.RewardEnsemble(
        venv.observation_space,
        venv.action_space,
        members=[
            reward_nets.BasicRewardNet(venv.observation_space, venv.action_space)
            for _ in range(2)
        ],
    )
    preference_model = preference_comparisons.PreferenceModel(
        model=reward_net,
        noise_prob=0.1,
        discount_factor=0.9,
        threshold=50,
    )

    fragmenter = preference_comparisons.ActiveSelectionFragmenter(
        preference_model=preference_model,
        base_fragmenter=random_fragmenter,
        fragment_sample_factor=2,
        uncertainty_on=uncertainty_on,
        custom_logger=custom_logger,
    )

    preference_model = preference_comparisons.PreferenceModel(
        model=reward_net,
        noise_prob=0.1,
        discount_factor=0.9,
        threshold=50,
    )
    loss = preference_comparisons.CrossEntropyRewardLoss()

    reward_trainer = preference_comparisons.EnsembleTrainer(
        preference_model,
        loss,
        rng=rng,
    )

    main_trainer = preference_comparisons.PreferenceComparisons(
        agent_trainer,
        reward_net,
        num_iterations=2,
        transition_oversampling=2,
        fragment_length=2,
        fragmenter=fragmenter,
        rng=rng,
        reward_trainer=reward_trainer,
        custom_logger=custom_logger,
    )
    main_trainer.train(100, 10)


@pytest.fixture(scope="module")
def interval_param_scaler() -> updaters.IntervalParamScaler:
    return updaters.IntervalParamScaler(
        scaling_factor=0.1,
        tolerable_interval=(1.1, 1.5),
    )


def test_reward_trainer_regularization_no_crash(
        agent_trainer,
        venv,
        random_fragmenter,
        custom_logger,
        preference_model,
        interval_param_scaler,
        rng,
):
    reward_net = reward_nets.BasicRewardNet(venv.observation_space, venv.action_space)
    loss = preference_comparisons.CrossEntropyRewardLoss()
    initial_lambda = 0.1
    regularizer_factory = regularizers.LpRegularizer.create(
        initial_lambda=initial_lambda,
        val_split=0.2,
        lambda_updater=interval_param_scaler,
        p=2,
    )
    reward_trainer = preference_comparisons.BasicRewardTrainer(
        preference_model,
        loss,
        regularizer_factory=regularizer_factory,
        custom_logger=custom_logger,
        rng=rng,
    )

    main_trainer = preference_comparisons.PreferenceComparisons(
        agent_trainer,
        reward_net,
        num_iterations=2,
        transition_oversampling=2,
        fragment_length=2,
        fragmenter=random_fragmenter,
        reward_trainer=reward_trainer,
        custom_logger=custom_logger,
        rng=rng,
    )
    main_trainer.train(50, 50)


def test_reward_trainer_regularization_raises(
        agent_trainer,
        venv,
        random_fragmenter,
        custom_logger,
        preference_model,
        interval_param_scaler,
        rng,
):
    reward_net = reward_nets.BasicRewardNet(venv.observation_space, venv.action_space)
    loss = preference_comparisons.CrossEntropyRewardLoss()
    initial_lambda = 0.1
    regularizer_factory = regularizers.LpRegularizer.create(
        initial_lambda=initial_lambda,
        val_split=0.2,
        lambda_updater=interval_param_scaler,
        p=2,
    )
    reward_trainer = preference_comparisons.BasicRewardTrainer(
        preference_model,
        loss,
        regularizer_factory=regularizer_factory,
        custom_logger=custom_logger,
        rng=rng,
    )

    main_trainer = preference_comparisons.PreferenceComparisons(
        agent_trainer,
        reward_net,
        num_iterations=2,
        transition_oversampling=2,
        fragment_length=2,
        fragmenter=random_fragmenter,
        reward_trainer=reward_trainer,
        custom_logger=custom_logger,
        rng=rng,
    )
    with pytest.raises(
            ValueError,
            match="Not enough data samples to split " "into training and validation.*",
    ):
        main_trainer.train(100, 10)


@pytest.fixture
def ensemble_preference_model(venv) -> preference_comparisons.PreferenceModel:
    reward_net = reward_nets.RewardEnsemble(
        venv.observation_space,
        venv.action_space,
        members=[
            reward_nets.BasicRewardNet(venv.observation_space, venv.action_space)
            for _ in range(2)
        ],
    )
    return preference_comparisons.PreferenceModel(
        model=reward_net,
        noise_prob=0.1,
        discount_factor=0.9,
        threshold=50,
    )


@pytest.fixture
def preference_model(venv) -> preference_comparisons.PreferenceModel:
    reward_net = reward_nets.BasicRewardNet(venv.observation_space, venv.action_space)
    return preference_comparisons.PreferenceModel(
        model=reward_net,
        noise_prob=0.1,
        discount_factor=0.9,
        threshold=50,
    )


def test_active_fragmenter_uncertainty_on_not_supported_error(
        ensemble_preference_model,
        random_fragmenter,
):
    re_match = r".* not supported\.\n\s+`uncertainty_on` should be from .*"
    with pytest.raises(ValueError, match=re_match):
        preference_comparisons.ActiveSelectionFragmenter(
            preference_model=ensemble_preference_model,
            base_fragmenter=random_fragmenter,
            fragment_sample_factor=2,
            uncertainty_on="uncertainty_on",
        )

    with pytest.raises(ValueError, match=re_match):
        fragmenter = preference_comparisons.ActiveSelectionFragmenter(
            preference_model=ensemble_preference_model,
            base_fragmenter=random_fragmenter,
            fragment_sample_factor=2,
            uncertainty_on="logit",
        )
        fragmenter._uncertainty_on = "uncertainty_on"
        members = ensemble_preference_model.model.num_members
        fragmenter.variance_estimate(th.rand(10, members), th.rand(10, members))


def test_active_selection_raises_error_when_initialized_without_an_ensemble(
        preference_model,
        random_fragmenter,
):
    with pytest.raises(
            ValueError,
            match=r"PreferenceModel not wrapped over an ensemble.*",
    ):
        preference_comparisons.ActiveSelectionFragmenter(
            preference_model=preference_model,
            base_fragmenter=random_fragmenter,
            fragment_sample_factor=2,
            uncertainty_on="logit",
        )


def test_agent_trainer_populates_buffer(agent_trainer):
    agent_trainer.train(steps=1)
    assert agent_trainer.buffering_wrapper.n_transitions > 0


def test_agent_trainer_sample(venv, agent_trainer):
    trajectories = agent_trainer.sample(2)
    assert len(trajectories) > 0
    assert all(
        trajectory.obs.shape[1:] == venv.observation_space.shape
        for trajectory in trajectories
    )


def test_agent_trainer_sample_image_observations(rng):
    """Test `AgentTrainer.sample()` in an image environment.

    SB3 algorithms may rearrange the channel dimension in environments with image
    observations, but `sample()` should return observations matching the original
    environment.

    Args:
        rng: Random number generator (with a fixed seed).
    """
    venv = DummyVecEnv([lambda: FakeImageEnv()])
    reward_net = reward_nets.BasicRewardNet(venv.observation_space, venv.action_space)
    agent = stable_baselines3.PPO(
        "MlpPolicy",
        venv,
        n_epochs=1,
        batch_size=2,
        n_steps=10,
    )
    agent_trainer = preference_comparisons.AgentTrainer(
        agent,
        reward_net,
        venv,
        exploration_frac=0.5,
        rng=rng,
    )
    trajectories = agent_trainer.sample(2)
    assert len(trajectories) > 0
    assert all(
        trajectory.obs.shape[1:] == venv.observation_space.shape
        for trajectory in trajectories
    )


class ActionIsRewardEnv(gym.Env):
    """Two step environment where the reward is the sum of actions."""

    def __init__(self):
        """Initialize environment."""
        super().__init__()
        self.action_space = spaces.Discrete(50)
        self.observation_space = gym.spaces.Box(np.array([0]), np.array([1]))
        self.steps = 0

    def step(self, action):
        obs = np.array([0])
        reward = action
        # Some algorithms expect trajectories at least two steps long,
        # so we allow two steps to be taken.
        done = self.steps > 0
        info = {}
        self.steps += 1
        return obs, reward, done, info

    def reset(self):
        self.steps = 0.0
        return np.array([0])


@pytest.fixture
def action_is_reward_venv():
    return DummyVecEnv(
        [ActionIsRewardEnv],
    )


@pytest.fixture
def action_is_reward_agent(action_is_reward_venv, rng):
    return stable_baselines3.PPO(
        "MlpPolicy",
        action_is_reward_venv,
        n_epochs=1,
        batch_size=2,
        n_steps=10,
    )


def basic_reward_trainer(venv, rng):
    loss = preference_comparisons.CrossEntropyRewardLoss()
    reward_net = reward_nets.BasicRewardNet(
        venv.observation_space,
        venv.action_space,
    )
    preference_model = preference_comparisons.PreferenceModel(
        model=reward_net,
        noise_prob=0.1,
        discount_factor=0.9,
        threshold=50,
    )
    return preference_comparisons.BasicRewardTrainer(
        preference_model,
        loss,
        rng=rng,
        lr=1e-4,
    )


def ensemble_reward_trainer(venv, rng):
    loss = preference_comparisons.CrossEntropyRewardLoss()
    reward_net = reward_nets.RewardEnsemble(
        venv.observation_space,
        venv.action_space,
        members=[
            reward_nets.BasicRewardNet(
                venv.observation_space,
                venv.action_space,
            )
            for _ in range(3)
        ],
    )
    preference_model = preference_comparisons.PreferenceModel(
        model=reward_net,
        noise_prob=0.1,
        discount_factor=0.9,
        threshold=50,
    )
    return preference_comparisons.EnsembleTrainer(
        preference_model,
        loss,
        rng=rng,
        lr=1e-4,
    )


@pytest.mark.parametrize(
    "action_is_reward_trainer_func",
    [basic_reward_trainer, ensemble_reward_trainer],
)
def test_that_trainer_improves(
        action_is_reward_venv,
        action_is_reward_agent,
        action_is_reward_trainer_func,
        random_fragmenter,
        custom_logger,
        rng,
):
    """Tests that training improves performance of the reward network and agent."""
    action_is_reward_trainer = action_is_reward_trainer_func(action_is_reward_venv, rng)
    agent_trainer = preference_comparisons.AgentTrainer(
        action_is_reward_agent,
        action_is_reward_trainer._preference_model.model,
        action_is_reward_venv,
        rng,
    )

    main_trainer = preference_comparisons.PreferenceComparisons(
        agent_trainer,
        action_is_reward_trainer._preference_model.model,
        num_iterations=2,
        transition_oversampling=2,
        fragment_length=2,
        fragmenter=random_fragmenter,
        rng=rng,
        reward_trainer=action_is_reward_trainer,
        custom_logger=custom_logger,
    )

    # Get initial agent performance
    novice_agent_rewards, _ = evaluation.evaluate_policy(
        agent_trainer.algorithm.policy,
        action_is_reward_venv,
        15,
        return_episode_rewards=True,
    )

    # Train for a short period of time, and then again.
    # We expect the reward network to have a better estimate of the reward
    # after this training, and thus `later_rewards` should have lower loss.
    first_reward_network_stats = main_trainer.train(20, 20)

    later_reward_network_stats = main_trainer.train(1000, 20)
    assert (
            first_reward_network_stats["reward_loss"]
            > later_reward_network_stats["reward_loss"]
    )

    # The agent should have also improved
    trained_agent_rewards, _ = evaluation.evaluate_policy(
        agent_trainer.algorithm.policy,
        action_is_reward_venv,
        15,
        return_episode_rewards=True,
    )

    assert np.mean(trained_agent_rewards) > np.mean(novice_agent_rewards)


def test_returns_query_dict_from_query_sequence_with_correct_length():
    querent = PreferenceQuerent()
    query_sequence = [Mock()]
    query_dict = querent(query_sequence)
    assert len(query_dict) == len(query_sequence)


def test_returned_queries_have_uuid():
    querent = PreferenceQuerent()
    query_dict = querent([Mock()])

    try:
        key = list(query_dict.keys())[0]
        uuid.UUID(key, version=4)
    except ValueError:
        pytest.fail()


def test_sends_put_request_for_each_query(requests_mock):
    address = "https://test.de"
    querent = PrefCollectQuerent(pref_collect_address=address, video_output_dir="video")
    query_id = "1234"

    requests_mock.put(f"{address}/preferences/query/{query_id}")
    querent._query(query_id)

    assert requests_mock.last_request.method == "PUT"
    assert requests_mock.last_request.text == f'{{"uuid": "{query_id}"}}'


class ConcretePreferenceGatherer(PreferenceGatherer):

    def __call__(self) -> Tuple[np.ndarray, np.ndarray]:
        pass


def test_adds_queries_to_pending_queries():
    gatherer = ConcretePreferenceGatherer()
    query_id = "id"
    queries = {query_id: Mock()}

    gatherer.add(new_queries=queries)
    assert query_id in list(gatherer.pending_queries.keys())


def test_clears_pending_queries(trajectory_with_rew):
    gatherer = SyntheticGatherer(sample=False)

    queries = {"id": (trajectory_with_rew, trajectory_with_rew)}
    gatherer.add(new_queries=queries)

    gatherer()

    assert len(gatherer.pending_queries) == 0


def test_returns_none_for_unanswered_query(requests_mock):
    address = "https://test.de"
    query_id = "1234"
    answer = None

    gatherer = PrefCollectGatherer(pref_collect_address=address)

    requests_mock.get(f"{address}/preferences/query/{query_id}", json={"query_id": query_id, "label": answer})

    preference = gatherer._gather_preference(query_id)

    assert preference is answer


def test_returns_preference_for_answered_query(requests_mock):
    address = "https://test.de"
    query_id = "1234"
    answer = 1.0

    gatherer = PrefCollectGatherer(pref_collect_address=address)

    requests_mock.get(f"{address}/preferences/query/{query_id}", json={"query_id": query_id, "label": answer})

    preference = gatherer._gather_preference(query_id)

    assert preference == answer


def test_keeps_pending_query_for_unanswered_query():
    gatherer = PrefCollectGatherer(pref_collect_address="https://test.de", wait_for_user=False)
    gatherer._gather_preference = MagicMock(return_value=None)
    gatherer.pending_queries = {"1234": Mock()}

    pending_queries_pre = gatherer.pending_queries.copy()
    gatherer()

    assert pending_queries_pre == gatherer.pending_queries


def test_delete_pending_query_for_answered_query():
    gatherer = PrefCollectGatherer(pref_collect_address="https://test.de", wait_for_user=False)
    gatherer._gather_preferences = MagicMock(return_value=None)

    pending_queries_pre = gatherer.pending_queries.copy()
    gatherer()

    assert pending_queries_pre == gatherer.pending_queries



