"""Tests `imitation.rewards.reward_nets` and `imitation.rewards.serialize`."""

import logging
import numbers
import os
import tempfile
from typing import Tuple
from unittest import mock

import gym
import numpy as np
import pytest
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv

import imitation.testing.reward_nets as testing_reward_nets
from imitation.data import rollout
from imitation.policies import base
from imitation.rewards import reward_nets, serialize
from imitation.util import networks, util


def _potential(x):
    # _potential is never actually called in the tests: we just need a dummy
    # potential to be able to construct shaped reward networks.
    return th.zeros(x.shape[0], device=x.device)  # pragma: no cover


ENVS = ["FrozenLake-v1", "CartPole-v1", "Pendulum-v1"]
DESERIALIZATION_TYPES = [
    "zero",
    "RewardNet_normalized",
    "RewardNet_unnormalized",
    "RewardNet_shaped",
    "RewardNet_unshaped",
]


# Reward net classes, allowed kwargs
MAKE_REWARD_NET = [
    reward_nets.BasicRewardNet,
    reward_nets.BasicShapedRewardNet,
    testing_reward_nets.make_ensemble,
]


MAKE_BASIC_REWARD_NET_WRAPPERS = [
    lambda base: reward_nets.ShapedRewardNet(base, _potential, 0.99),
    lambda base: reward_nets.NormalizedRewardNet(base, networks.RunningNorm),
]

REWARD_NET_KWARGS = [
    {},
    {"normalize_input_layer": networks.RunningNorm},
]

NORMALIZE_OUTPUT_LAYER = [
    None,
    networks.RunningNorm,
]


NumpyTransitions = Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]


@pytest.fixture
def numpy_transitions() -> NumpyTransitions:
    """A batch of states, actions, next_states, and dones as np.ndarrays for Env2D."""
    return (
        np.zeros((10, 5, 5)),
        np.zeros((10, 1), dtype=int),
        np.zeros((10, 5, 5)),
        np.zeros((10,), dtype=bool),
    )


TorchTransitions = Tuple[th.Tensor, th.Tensor, th.Tensor, th.Tensor]


@pytest.fixture
def torch_transitions() -> TorchTransitions:
    """A batch of states, actions, next_states, and dones as th.Tensors for Env2D."""
    return (
        th.zeros((10, 5, 5)),
        th.zeros((10, 1), dtype=int),
        th.zeros((10, 5, 5)),
        th.zeros((10,), dtype=bool),
    )


@pytest.mark.parametrize("env_name", ENVS)
@pytest.mark.parametrize("reward_net_cls", MAKE_REWARD_NET)
@pytest.mark.parametrize("reward_net_kwargs", REWARD_NET_KWARGS)
@pytest.mark.parametrize("normalize_output_layer", NORMALIZE_OUTPUT_LAYER)
def test_init_no_crash(
    env_name,
    reward_net_cls,
    reward_net_kwargs,
    normalize_output_layer,
):
    env = gym.make(env_name)

    reward_net = reward_net_cls(
        env.observation_space,
        env.action_space,
        **reward_net_kwargs,
    )

    if normalize_output_layer:
        reward_net = reward_nets.NormalizedRewardNet(
            reward_net,
            normalize_output_layer,
        )


def _sample(space, n):
    return np.array([space.sample() for _ in range(n)])


def _make_env_and_save_reward_net(env_name, reward_type, tmpdir):
    venv = util.make_vec_env(env_name, n_envs=1, parallel=False)
    save_path = os.path.join(tmpdir, "norm_reward.pt")

    assert reward_type in [
        "zero",
        "RewardNet_normalized",
        "RewardNet_unnormalized",
        "RewardNet_shaped",
        "RewardNet_unshaped",
    ], f"Reward net type {reward_type} not supported by this helper."

    if reward_type == "zero":
        return venv, save_path

    net = reward_nets.BasicRewardNet(venv.observation_space, venv.action_space)

    if reward_type == "RewardNet_normalized":
        net = reward_nets.NormalizedRewardNet(net, networks.RunningNorm)
    elif reward_type == "RewardNet_shaped":
        net = reward_nets.ShapedRewardNet(net, _potential, discount_factor=0.99)
    elif reward_type in ["RewardNet_unshaped", "RewardNet_unnormalized"]:
        pass

    th.save(net, save_path)
    return venv, save_path


@pytest.mark.parametrize("env_name", ENVS)
@pytest.mark.parametrize("reward_type", DESERIALIZATION_TYPES)
def test_reward_valid(env_name, reward_type, tmpdir):
    """Test output of reward function is appropriate shape and type."""
    venv = util.make_vec_env(env_name, n_envs=1, parallel=False)
    venv, tmppath = _make_env_and_save_reward_net(env_name, reward_type, tmpdir)

    TRAJECTORY_LEN = 10
    obs = _sample(venv.observation_space, TRAJECTORY_LEN)
    actions = _sample(venv.action_space, TRAJECTORY_LEN)
    next_obs = _sample(venv.observation_space, TRAJECTORY_LEN)
    steps = np.arange(0, TRAJECTORY_LEN)

    reward_fn = serialize.load_reward(reward_type, tmppath, venv)
    pred_reward = reward_fn(obs, actions, next_obs, steps)

    assert pred_reward.shape == (TRAJECTORY_LEN,)
    assert isinstance(pred_reward[0], numbers.Number)


def test_wrappers_default_to_passing_on_method_calls_to_base(
    numpy_transitions: NumpyTransitions,
    torch_transitions: TorchTransitions,
):
    base = mock.MagicMock()
    wrapper = reward_nets.RewardNetWrapper(base)
    # Check method calls
    for attr, call_with, return_value in [
        ("forward", torch_transitions, th.zeros(10)),
        ("predict_th", numpy_transitions, np.zeros(10)),
        ("predict", numpy_transitions, np.zeros(10)),
        ("predict_processed", numpy_transitions, np.zeros(10)),
        ("preprocess", numpy_transitions, torch_transitions),
    ]:
        setattr(base, attr, mock.MagicMock(return_value=return_value))
        assert getattr(wrapper, attr)(*call_with) is return_value
        getattr(base, attr).assert_called_once_with(*call_with)

    # Check property lookups
    return_value = th.device("cpu")
    base.device = mock.PropertyMock(return_value=return_value)
    assert wrapper.device is base.device

    return_value = th.float32
    base.dtype = mock.PropertyMock(return_value=return_value)
    assert wrapper.dtype is base.dtype


def test_strip_wrappers_basic():
    venv = util.make_vec_env("FrozenLake-v1", n_envs=1, parallel=False)
    net = reward_nets.BasicRewardNet(venv.observation_space, venv.action_space)
    net = reward_nets.NormalizedRewardNet(net, networks.RunningNorm)
    net = serialize._strip_wrappers(
        net,
        wrapper_types=[reward_nets.NormalizedRewardNet],
    )
    assert isinstance(net, reward_nets.BasicRewardNet)
    # This removing a wrapper from an unwrapped reward net should do nothing
    net = serialize._strip_wrappers(net, wrapper_types=[reward_nets.ShapedRewardNet])
    assert isinstance(net, reward_nets.BasicRewardNet)


def test_strip_wrappers_complex():
    venv = util.make_vec_env("FrozenLake-v1", n_envs=1, parallel=False)
    net = reward_nets.BasicRewardNet(venv.observation_space, venv.action_space)
    net = reward_nets.ShapedRewardNet(net, _potential, discount_factor=0.99)
    net = reward_nets.NormalizedRewardNet(net, networks.RunningNorm)
    # Removing in incorrect order should do nothing
    net = serialize._strip_wrappers(
        net,
        wrapper_types=[reward_nets.ShapedRewardNet, reward_nets.NormalizedRewardNet],
    )

    assert isinstance(net, reward_nets.NormalizedRewardNet)
    assert isinstance(net.base, reward_nets.ShapedRewardNet)
    # Correct order should work
    net = serialize._strip_wrappers(
        net,
        wrapper_types=[reward_nets.NormalizedRewardNet, reward_nets.ShapedRewardNet],
    )
    assert isinstance(net, reward_nets.BasicRewardNet)


def test_validate_wrapper_structure():
    env = gym.make("FrozenLake-v1")

    class RewardNetA(reward_nets.RewardNet):
        def forward(*args):
            ...  # pragma: no cover

    class WrapperB(reward_nets.RewardNetWrapper):
        def forward(*args):
            ...  # pragma: no cover

    reward_net = RewardNetA(env.action_space, env.observation_space)
    reward_net = WrapperB(reward_net)

    assert isinstance(reward_net.base, RewardNetA)

    # This should not raise a type error
    serialize._validate_wrapper_structure(reward_net, {(WrapperB, RewardNetA)})

    raises_error_ctxmgr = pytest.raises(
        TypeError,
        match=r"Wrapper structure should match \[.*\] but found \[.*\]",
    )

    # The top level wrapper is an instance of WrapperB this should raise a type error
    with raises_error_ctxmgr:
        serialize._validate_wrapper_structure(reward_net, {(RewardNetA,)})

    # Reward net is not wrapped at all this should raise a type error.
    with raises_error_ctxmgr:
        serialize._validate_wrapper_structure(
            RewardNetA(env.action_space, env.observation_space),
            {(WrapperB,)},
        )

    # The prefix is longer then set of wrappers
    with raises_error_ctxmgr:
        serialize._validate_wrapper_structure(
            reward_net,
            {(WrapperB, RewardNetA, WrapperB)},
        )

    # This should not raise a type error since one of the prefixes matches
    serialize._validate_wrapper_structure(
        reward_net,
        {(WrapperB, RewardNetA), (RewardNetA,)},
    )

    # This should raise a type error since none the prefix is in the incorrect order
    with raises_error_ctxmgr:
        serialize._validate_wrapper_structure(reward_net, {(RewardNetA, WrapperB)})


@pytest.mark.parametrize("env_name", ENVS)
def test_cant_load_unnorm_as_norm(env_name, tmpdir):
    venv, tmppath = _make_env_and_save_reward_net(
        env_name,
        "RewardNet_unnormalized",
        tmpdir,
    )
    with pytest.raises(TypeError):
        serialize.load_reward("RewardNet_normalized", tmppath, venv)


@pytest.mark.parametrize("env_name", ENVS)
@pytest.mark.parametrize("net_cls", MAKE_REWARD_NET)
@pytest.mark.parametrize("normalize_rewards", [True, False])
def test_serialize_identity(
    env_name,
    net_cls,
    normalize_rewards,
    tmpdir,
):
    """Does output of deserialized reward network match that of original?"""
    logging.info(f"Testing {net_cls}")

    venv = util.make_vec_env(env_name, n_envs=1, parallel=False)
    original = net_cls(venv.observation_space, venv.action_space)
    if normalize_rewards:
        original = reward_nets.NormalizedRewardNet(original, networks.RunningNorm)
    random = base.RandomPolicy(venv.observation_space, venv.action_space)

    tmppath = os.path.join(tmpdir, "reward.pt")
    th.save(original, tmppath)
    loaded = th.load(tmppath)

    assert original.observation_space == loaded.observation_space
    assert original.action_space == loaded.action_space

    transitions = rollout.generate_transitions(random, venv, n_timesteps=100)

    if isinstance(original, reward_nets.NormalizedRewardNet):
        wrapped_rew_fn = serialize.load_reward("RewardNet_normalized", tmppath, venv)
        unwrapped_rew_fn = serialize.load_reward(
            "RewardNet_unnormalized",
            tmppath,
            venv,
        )
    elif isinstance(original, reward_nets.ShapedRewardNet):
        unwrapped_rew_fn = serialize.load_reward("RewardNet_unshaped", tmppath, venv)
        wrapped_rew_fn = serialize.load_reward("RewardNet_shaped", tmppath, venv)
    else:
        unwrapped_rew_fn = serialize.load_reward("RewardNet_unshaped", tmppath, venv)
        wrapped_rew_fn = unwrapped_rew_fn

    rewards = {
        "wrapped": [],
        "unwrapped": [],
    }
    for net in [original, loaded]:
        trans_args = (
            transitions.obs,
            transitions.acts,
            transitions.next_obs,
            transitions.dones,
        )
        rewards["wrapped"].append(net.predict(*trans_args))
        if hasattr(net, "base"):
            rewards["unwrapped"].append(net.base.predict(*trans_args))
        else:
            rewards["unwrapped"].append(net.predict(*trans_args))

    args = (
        transitions.obs,
        transitions.acts,
        transitions.next_obs,
        transitions.dones,
    )
    rewards["wrapped"].append(wrapped_rew_fn(*args))
    rewards["unwrapped"].append(unwrapped_rew_fn(*args))

    for key, predictions in rewards.items():
        assert len(predictions) == 3
        assert np.allclose(predictions[0], predictions[1])
        assert np.allclose(predictions[0], predictions[2])


class Env2D(gym.Env):
    """Mock environment with 2D observations."""

    def __init__(self):
        """Builds `Env2D`."""
        super().__init__()
        self.observation_space = gym.spaces.Box(shape=(5, 5), low=-1.0, high=1.0)
        self.action_space = gym.spaces.Discrete(2)

    def step(self, action):
        obs = self.observation_space.sample()
        rew = 0.0
        done = False
        info = {}
        return obs, rew, done, info

    def reset(self):
        return self.observation_space.sample()


def test_potential_net_2d_obs():
    """Test potential net can do forward-prop with 2D observation.

    This is a regression test for a problem identified Eric. Previously, reward
    nets would not properly flatten N-dimensional states before passing them to
    potential networks, leading to shape mismatches.
    """
    # instantiate environment & get batch observations, actions, etc.
    env = Env2D()
    obs = env.reset()
    action = env.action_space.sample()
    next_obs, _, done, _ = env.step(action)
    obs_b = obs[None]
    action_b = np.array([action], dtype="int")
    next_obs_b = next_obs[None]
    done_b = np.array([done], dtype="bool")

    net = reward_nets.BasicShapedRewardNet(
        env.observation_space,
        env.action_space,
    )
    rew_batch = net.predict(obs_b, action_b, next_obs_b, done_b)
    assert rew_batch.shape == (1,)


class MockRewardNet(reward_nets.RewardNet):
    """A mock reward net for testing."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        value: float = 0.0,
    ):
        """Create mock reward.

        Args:
            observation_space: observation space of the env
            action_space: action space of the env
            value: The reward to always return. Defaults to 0.0.
        """
        super().__init__(observation_space, action_space)
        self.value = value

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        batch_size = state.shape[0]
        return th.full(
            (batch_size,),
            fill_value=self.value,
            dtype=th.float32,
            device=state.device,
        )


@pytest.fixture
def env_2d() -> Env2D:
    """An instance of Env2d."""
    return Env2D()


def test_ensemble_errors_if_there_are_too_few_members(env_2d):
    for num_members in range(2):
        with pytest.raises(ValueError):
            reward_nets.RewardEnsemble(
                env_2d.observation_space,
                env_2d.action_space,
                members=[
                    MockRewardNet(env_2d.observation_space, env_2d.action_space)
                    for _ in range(num_members)
                ],
            )


@pytest.fixture
def two_ensemble(env_2d) -> reward_nets.RewardEnsemble:
    """A simple reward ensemble made up of two mock reward nets."""
    return reward_nets.RewardEnsemble(
        env_2d.observation_space,
        env_2d.action_space,
        members=[
            MockRewardNet(env_2d.observation_space, env_2d.action_space)
            for _ in range(2)
        ],
    )


def test_reward_ensemble_predict_reward_moments(
    two_ensemble: reward_nets.RewardEnsemble,
    numpy_transitions: NumpyTransitions,
):
    # Test that the calculation of mean and variance is correct
    two_ensemble.members[0].value = 0
    two_ensemble.members[1].value = 0
    mean, var = two_ensemble.predict_reward_moments(*numpy_transitions)
    assert np.isclose(mean, 0).all()
    assert np.isclose(var, 0).all()
    two_ensemble.members[0].value = 3
    two_ensemble.members[1].value = -1
    mean, var = two_ensemble.predict_reward_moments(*numpy_transitions)
    assert np.isclose(mean, 1).all()
    assert np.isclose(var, 8).all()  # note we are using the unbiased variance estimator
    # Test that ensemble calls members correctly
    two_ensemble.members[0].forward = mock.MagicMock(return_value=th.zeros(10))
    mean, var = two_ensemble.predict_reward_moments(*numpy_transitions)
    two_ensemble.members[0].forward.assert_called_once()


def test_ensemble_members_have_different_parameters(env_2d):
    ensemble = testing_reward_nets.make_ensemble(
        env_2d.observation_space,
        env_2d.action_space,
    )

    assert not th.allclose(
        next(ensemble.members[0].parameters()),
        next(ensemble.members[1].parameters()),
    )


def test_add_std_wrapper_raises_error_when_wrapping_wrong_type(env_2d):
    mock_env = MockRewardNet(env_2d.observation_space, env_2d.action_space)
    assert not isinstance(mock_env, reward_nets.RewardNetWithVariance)
    with pytest.raises(TypeError):
        reward_nets.AddSTDRewardWrapper(mock_env, default_alpha=0.1)


def test_add_std_reward_wrapper(
    two_ensemble: reward_nets.RewardEnsemble,
    numpy_transitions: NumpyTransitions,
):
    two_ensemble.members[0].value = 3
    two_ensemble.members[1].value = -1
    reward_fn = reward_nets.AddSTDRewardWrapper(two_ensemble, default_alpha=0.1)
    rewards = reward_fn.predict_processed(*numpy_transitions)
    assert np.allclose(rewards, 1 + 0.1 * np.sqrt(8))
    # test overriding in predict processed works correctly
    rewards = reward_fn.predict_processed(*numpy_transitions, alpha=-0.5)
    assert np.allclose(rewards, 1 - 0.5 * np.sqrt(8))


@pytest.mark.parametrize("make_wrapper", MAKE_BASIC_REWARD_NET_WRAPPERS)
def test_wrappers_pass_on_kwargs(
    make_wrapper: reward_nets.RewardNetWrapper,
    env_2d: Env2D,
    numpy_transitions: NumpyTransitions,
):
    basic_reward_net = reward_nets.BasicRewardNet(
        env_2d.observation_space,
        env_2d.action_space,
    )
    basic_reward_net.predict_processed = mock.Mock(return_value=np.zeros((10,)))
    wrapped_reward_net = make_wrapper(
        basic_reward_net,
    )
    wrapped_reward_net.predict_processed(
        *numpy_transitions,
        foobar=42,
    )
    basic_reward_net.predict_processed.assert_called_once_with(
        *numpy_transitions,
        foobar=42,
    )


def test_load_reward_passes_along_alpha_to_add_std_wrappers_predict_processed_method(
    env_2d: Env2D,
    two_ensemble: reward_nets.RewardEnsemble,
    numpy_transitions: NumpyTransitions,
):
    """Kwargs passed to load_reward are passed along to predict_processed."""
    two_ensemble.members[0].value = 3
    two_ensemble.members[1].value = -1
    reward_net = reward_nets.AddSTDRewardWrapper(two_ensemble, default_alpha=0)
    with tempfile.TemporaryDirectory() as tmp_dir:
        net_path = os.path.join(tmp_dir, "reward_net.pkl")
        th.save(reward_net, os.path.join(net_path))
        new_alpha = -0.5
        reward_fn = serialize.load_reward(
            "RewardNet_std_added",
            net_path,
            env_2d,
            alpha=new_alpha,
        )
        rewards = reward_fn(*numpy_transitions)
        assert np.allclose(rewards, 1 + new_alpha * np.sqrt(8))


@pytest.mark.parametrize("env_name", ENVS)
def test_device_for_parameterless_model(env_name):
    class ParameterlessNet(reward_nets.RewardNet):
        def forward(self):
            """Dummy function to avoid abstractmethod complaints."""

    env = gym.make(env_name)
    net = ParameterlessNet(env.observation_space, env.action_space)
    assert net.device == th.device("cpu")


@pytest.mark.parametrize("normalize_input_layer", [None, networks.RunningNorm])
def test_training_regression(normalize_input_layer):
    """Test reward_net normalization by training a regression model."""
    venv = DummyVecEnv([lambda: gym.make("CartPole-v0")] * 2)
    reward_net = reward_nets.BasicRewardNet(
        venv.observation_space,
        venv.action_space,
        normalize_input_layer=normalize_input_layer,
    )
    norm_rew_net = reward_nets.NormalizedRewardNet(
        reward_net,
        normalize_output_layer=networks.RunningNorm,
    )

    # Construct a loss function and an Optimizer.
    criterion = th.nn.MSELoss(reduction="sum")
    optimizer = th.optim.SGD(norm_rew_net.parameters(), lr=1e-6)

    # Getting transitions from a random policy
    random = base.RandomPolicy(venv.observation_space, venv.action_space)
    for _ in range(2):
        transitions = rollout.generate_transitions(random, venv, n_timesteps=100)
        trans_args = (
            transitions.obs,
            transitions.acts,
            transitions.next_obs,
            transitions.dones,
        )
        trans_args_th = norm_rew_net.preprocess(*trans_args)
        rews_th = norm_rew_net(*trans_args_th)
        rews = rews_th.detach().cpu().numpy().flatten()

        # Compute and print loss
        loss = criterion(
            util.safe_to_tensor(transitions.rews).to(norm_rew_net.device),
            rews_th,
        )

        # Get rewards from norm_rew_net.predict() and norm_rew_net.predict_processed()
        rews_predict = norm_rew_net.predict(*trans_args)
        rews_processed = norm_rew_net.predict_processed(*trans_args)

        # norm_rew_net() and norm_rew_net.predict() don't pass the reward through the
        # normalization layer, so the values of `rews` and `rews_predict` are identical
        assert (rews == rews_predict).all()
        # norm_rew_net.predict_processed() does normalize the reward
        assert not (rews_processed == rews_predict).all()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
