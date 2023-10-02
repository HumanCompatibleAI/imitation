"""Tests `imitation.rewards.reward_nets` and `imitation.rewards.serialize`."""

import functools
import logging
import os
import tempfile
from typing import Callable, Tuple
from unittest import mock

import gymnasium as gym
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
IMAGE_ENVS = ["Asteroids-v4"]
DESERIALIZATION_TYPES = [
    "zero",
    "RewardNet_normalized",
    "RewardNet_unnormalized",
    "RewardNet_shaped",
    "RewardNet_unshaped",
]


# Reward net classes, allowed kwargs
MAKE_REWARD_NET = [
    functools.partial(
        reward_nets.BasicRewardNet,
        hid_sizes=(3,),
    ),
    functools.partial(
        reward_nets.BasicShapedRewardNet,
        reward_hid_sizes=(3,),
        potential_hid_sizes=(3,),
    ),
    functools.partial(
        testing_reward_nets.make_ensemble,
        hid_sizes=(3,),
    ),
]
MAKE_IMAGE_REWARD_NET = [
    functools.partial(
        reward_nets.CnnRewardNet,
        hid_channels=(3, 3, 3),
        kernel_size=3,
        padding=0,
        stride=3,
    ),
]

MakePredictProcessedWrapper = Callable[
    [reward_nets.RewardNet],
    reward_nets.PredictProcessedWrapper,
]
MAKE_PREDICT_PROCESSED_WRAPPERS = [
    lambda base: reward_nets.NormalizedRewardNet(base, networks.RunningNorm),
]

MakeForwardWrapper = Callable[[reward_nets.RewardNet], reward_nets.ForwardWrapper]
MAKE_FORWARD_WRAPPERS = [
    lambda base: reward_nets.ShapedRewardNet(base, _potential, 0.99),
]

REWARD_NET_KWARGS = [
    {},
    {"normalize_input_layer": networks.RunningNorm},
    {"normalize_input_layer": networks.EMANorm},
    {"use_next_state": True, "dropout_prob": 0.3},
    {"use_done": True},
]

IMAGE_REWARD_NET_KWARGS = [
    {},
    {"use_next_state": True, "use_action": False},
    {"dropout_prob": 0.3},
    {"use_done": True, "use_action": True},
    {"use_done": True, "use_action": False},
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
        th.zeros((10, 1), dtype=th.int),
        th.zeros((10, 5, 5)),
        th.zeros((10,), dtype=th.bool),
    )


def _init_no_crash(env_name, reward_net_cls, reward_net_kwargs, normalize_output_layer):
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
    _init_no_crash(
        env_name,
        reward_net_cls,
        reward_net_kwargs,
        normalize_output_layer,
    )


@pytest.mark.parametrize("env_name", IMAGE_ENVS)
@pytest.mark.parametrize("reward_net_cls", MAKE_IMAGE_REWARD_NET)
@pytest.mark.parametrize("reward_net_kwargs", IMAGE_REWARD_NET_KWARGS)
@pytest.mark.parametrize("normalize_output_layer", NORMALIZE_OUTPUT_LAYER)
def test_image_init_no_crash(
    env_name,
    reward_net_cls,
    reward_net_kwargs,
    normalize_output_layer,
):
    _init_no_crash(
        env_name,
        reward_net_cls,
        reward_net_kwargs,
        normalize_output_layer,
    )


NOT_QUITE_IMAGE_SPACES = (
    gym.spaces.Box(0, 255, shape=(210, 160, 3), dtype=np.float32),
    gym.spaces.Box(0, 254, shape=(210, 160, 3), dtype=np.uint8),
    gym.spaces.Box(1, 255, shape=(210, 160, 3), dtype=np.uint8),
    gym.spaces.Box(0, 255, shape=(210, 160), dtype=np.uint8),
)


def test_cnn_reward_net_input_validation():
    atari_obs_space = gym.spaces.Box(0, 255, shape=(210, 160, 3), dtype=np.uint8)
    atari_action_space = gym.spaces.Discrete(14)

    with pytest.raises(ValueError, match="must take current or next state"):
        reward_nets.CnnRewardNet(
            atari_obs_space,
            atari_action_space,
            use_state=False,
            use_next_state=False,
        )

    for obs_space in NOT_QUITE_IMAGE_SPACES:
        with pytest.raises(ValueError, match="requires observations to be images"):
            reward_nets.CnnRewardNet(
                obs_space,
                atari_action_space,
            )

    illegal_act_space = gym.spaces.MultiDiscrete((2, 2))
    with pytest.raises(ValueError, match="can only use Discrete action spaces."):
        reward_nets.CnnRewardNet(
            atari_obs_space,
            illegal_act_space,
        )

    reward_nets.CnnRewardNet(
        atari_obs_space,
        illegal_act_space,
        use_action=False,
    )


def test_cnn_potential_input_validation():
    for obs_space in NOT_QUITE_IMAGE_SPACES:
        with pytest.raises(ValueError, match="must be given image inputs"):
            reward_nets.BasicPotentialCNN(
                obs_space,
                hid_sizes=(32,),
            )


@pytest.mark.parametrize("dimensions", (1, 3, 4, 5))
def test_cnn_transpose_input_validation(dimensions: int):
    shape = (2,) * dimensions
    tens = th.zeros(shape)

    if dimensions == 4:  # should succeed
        reward_nets.cnn_transpose(tens)
    else:  # should fail
        with pytest.raises(ValueError, match="Invalid input: "):
            reward_nets.cnn_transpose(tens)


def _sample(space, n):
    return np.array([space.sample() for _ in range(n)])


def _make_env_and_save_reward_net(env_name, reward_type, tmpdir, rng, is_image=False):
    venv = util.make_vec_env(
        env_name,
        n_envs=1,
        parallel=False,
        rng=rng,
    )
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

    net_cls = reward_nets.CnnRewardNet if is_image else reward_nets.BasicRewardNet
    net = net_cls(venv.observation_space, venv.action_space)

    if reward_type == "RewardNet_normalized":
        net = reward_nets.NormalizedRewardNet(net, networks.RunningNorm)
    elif reward_type == "RewardNet_shaped":
        pot_cls = (
            reward_nets.BasicPotentialMLP
            if not is_image
            else reward_nets.BasicPotentialCNN
        )
        potential = pot_cls(venv.observation_space, [8, 8])
        net = reward_nets.ShapedRewardNet(net, potential, discount_factor=0.99)
    elif reward_type in ["RewardNet_unshaped", "RewardNet_unnormalized"]:
        pass

    th.save(net, save_path)
    return venv, save_path


def _is_reward_valid(env_name, reward_type, tmpdir, rng, is_image):
    venv, tmppath = _make_env_and_save_reward_net(
        env_name,
        reward_type,
        tmpdir,
        rng,
        is_image=is_image,
    )

    TRAJECTORY_LEN = 10
    obs = _sample(venv.observation_space, TRAJECTORY_LEN)
    actions = _sample(venv.action_space, TRAJECTORY_LEN)
    next_obs = _sample(venv.observation_space, TRAJECTORY_LEN)
    steps = np.arange(0, TRAJECTORY_LEN)

    reward_fn = serialize.load_reward(reward_type, tmppath, venv)
    pred_reward = reward_fn(obs, actions, next_obs, steps)

    assert isinstance(pred_reward, np.ndarray)
    assert pred_reward.shape == (TRAJECTORY_LEN,)
    assert np.issubdtype(pred_reward.dtype, np.number)


@pytest.mark.parametrize("env_name", ENVS)
@pytest.mark.parametrize("reward_type", DESERIALIZATION_TYPES)
def test_reward_valid(env_name, reward_type, tmpdir, rng):
    """Test output of reward function is appropriate shape and type."""
    _is_reward_valid(env_name, reward_type, tmpdir, rng, is_image=False)


@pytest.mark.parametrize("env_name", IMAGE_ENVS)
@pytest.mark.parametrize("reward_type", DESERIALIZATION_TYPES)
def test_reward_valid_image(env_name, reward_type, tmpdir, rng):
    """Test output of reward function is appropriate shape and type."""
    _is_reward_valid(env_name, reward_type, tmpdir, rng, is_image=True)


def test_strip_wrappers_basic(rng):
    venv = util.make_vec_env(
        "FrozenLake-v1",
        n_envs=1,
        parallel=False,
        rng=rng,
    )
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


def test_strip_wrappers_image_basic(rng):
    venv = util.make_vec_env("Asteroids-v4", n_envs=1, parallel=False, rng=rng)
    net = reward_nets.CnnRewardNet(venv.observation_space, venv.action_space)
    net = reward_nets.NormalizedRewardNet(net, networks.RunningNorm)
    net = serialize._strip_wrappers(
        net,
        wrapper_types=[reward_nets.NormalizedRewardNet],
    )
    assert isinstance(net, reward_nets.CnnRewardNet)
    # This removing a wrapper from an unwrapped reward net should do nothing
    net = serialize._strip_wrappers(net, wrapper_types=[reward_nets.ShapedRewardNet])
    assert isinstance(net, reward_nets.CnnRewardNet)


def test_strip_wrappers_complex(rng):
    venv = util.make_vec_env(
        "FrozenLake-v1",
        n_envs=1,
        parallel=False,
        rng=rng,
    )
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


def test_strip_wrappers_image_complex(rng):
    venv = util.make_vec_env("Asteroids-v4", n_envs=1, parallel=False, rng=rng)
    net = reward_nets.CnnRewardNet(venv.observation_space, venv.action_space)
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
    assert isinstance(net, reward_nets.CnnRewardNet)


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
def test_cant_load_unnorm_as_norm(env_name, tmpdir, rng):
    venv, tmppath = _make_env_and_save_reward_net(
        env_name,
        "RewardNet_unnormalized",
        tmpdir,
        rng=rng,
    )
    with pytest.raises(TypeError):
        serialize.load_reward("RewardNet_normalized", tmppath, venv)


def _serialize_deserialize_identity(
    env_name,
    net_cls,
    net_kwargs,
    normalize_rewards,
    tmpdir,
    rng,
):
    """Does output of deserialized reward network match that of original?"""
    logging.info(f"Testing {net_cls}")

    ep_steps = 5

    venv = util.make_vec_env(
        env_name,
        n_envs=1,
        parallel=False,
        rng=rng,
        max_episode_steps=ep_steps + 1,
    )
    original = net_cls(venv.observation_space, venv.action_space, **net_kwargs)
    if normalize_rewards:
        original = reward_nets.NormalizedRewardNet(original, networks.RunningNorm)
    random = base.RandomPolicy(venv.observation_space, venv.action_space)

    tmppath = os.path.join(tmpdir, "reward.pt")
    th.save(original, tmppath)
    loaded = th.load(tmppath)

    assert original.observation_space == loaded.observation_space
    assert original.action_space == loaded.action_space

    transitions = rollout.generate_transitions(
        random,
        venv,
        n_timesteps=ep_steps,
        rng=rng,
    )

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


@pytest.mark.parametrize("env_name", ENVS)
@pytest.mark.parametrize("net_cls", MAKE_REWARD_NET)
@pytest.mark.parametrize("net_kwargs", REWARD_NET_KWARGS)
@pytest.mark.parametrize("normalize_rewards", [True, False])
def test_serialize_identity(
    env_name,
    net_cls,
    net_kwargs,
    normalize_rewards,
    tmpdir,
    rng,
):
    """Does output of deserialized reward MLP match that of original?"""
    _serialize_deserialize_identity(
        env_name,
        net_cls,
        net_kwargs,
        normalize_rewards,
        tmpdir,
        rng,
    )


@pytest.mark.parametrize("env_name", IMAGE_ENVS)
@pytest.mark.parametrize("net_cls", MAKE_IMAGE_REWARD_NET)
@pytest.mark.parametrize("net_kwargs", IMAGE_REWARD_NET_KWARGS)
@pytest.mark.parametrize("normalize_rewards", [True, False])
def test_serialize_identity_images(
    env_name,
    net_cls,
    net_kwargs,
    normalize_rewards,
    tmpdir,
    rng,
):
    """Does output of deserialized reward CNN match that of original?"""
    _serialize_deserialize_identity(
        env_name,
        net_cls,
        net_kwargs,
        normalize_rewards,
        tmpdir,
        rng,
    )


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
                    testing_reward_nets.MockRewardNet(
                        env_2d.observation_space,
                        env_2d.action_space,
                    )
                    for _ in range(num_members)
                ],
            )


@pytest.fixture
def zero_reward_net(env_2d) -> testing_reward_nets.MockRewardNet:
    return testing_reward_nets.MockRewardNet(
        env_2d.observation_space,
        env_2d.action_space,
        value=0,
    )


@pytest.fixture(params=MAKE_PREDICT_PROCESSED_WRAPPERS)
def predict_processed_wrapper(request, zero_reward_net):
    return request.param(zero_reward_net)


@pytest.fixture
def two_ensemble(env_2d) -> reward_nets.RewardEnsemble:
    """A simple reward ensemble made up of two mock reward nets."""
    return reward_nets.RewardEnsemble(
        env_2d.observation_space,
        env_2d.action_space,
        members=[
            testing_reward_nets.MockRewardNet(
                env_2d.observation_space,
                env_2d.action_space,
            )
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
    mock_env = testing_reward_nets.MockRewardNet(
        env_2d.observation_space,
        env_2d.action_space,
    )
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


def test_shaped_reward_net(
    zero_reward_net: testing_reward_nets.MockRewardNet,
    numpy_transitions: NumpyTransitions,
):
    def potential(x: th.Tensor):
        return th.full((x.shape[0],), 10, device=x.device)

    shaped = reward_nets.ShapedRewardNet(zero_reward_net, potential, 0.9)
    # We expect the shaped reward to be -1 since,
    # r'(s,a,s') = r(s,a,s') + \gamma \theta(s') - \theta(s) = (0) + (0.9)(10) - 10 = -1
    shaped_rew = th.full((10,), -1, dtype=th.float32)
    forward_args = shaped.preprocess(*numpy_transitions)
    assert th.allclose(shaped(*forward_args), shaped_rew)
    assert th.allclose(shaped.predict_th(*numpy_transitions), shaped_rew)
    assert np.allclose(shaped.predict(*numpy_transitions), shaped_rew.numpy())
    assert np.allclose(shaped.predict_processed(*numpy_transitions), shaped_rew.numpy())


@pytest.mark.parametrize("make_forward_wrapper", MAKE_FORWARD_WRAPPERS)
def test_forward_wrapper_cannot_be_applied_predict_processed_wrapper(
    predict_processed_wrapper: reward_nets.PredictProcessedWrapper,
    make_forward_wrapper: MakeForwardWrapper,
):
    with pytest.raises(
        ValueError,
        match=r"ForwardWrapper cannot be applied on top of PredictProcessedWrapper!",
    ):
        make_forward_wrapper(predict_processed_wrapper)


@pytest.mark.parametrize(
    "make_predict_processed_wrapper",
    MAKE_PREDICT_PROCESSED_WRAPPERS,
)
def test_predict_processed_wrappers_pass_on_kwargs(
    make_predict_processed_wrapper: MakePredictProcessedWrapper,
    zero_reward_net: testing_reward_nets.MockRewardNet,
    numpy_transitions: NumpyTransitions,
):
    zero_reward_net.predict_processed = mock.Mock(  # type: ignore[assignment]
        return_value=np.zeros((10,)),
    )
    wrapped_reward_net = make_predict_processed_wrapper(
        zero_reward_net,
    )
    wrapped_reward_net.predict_processed(
        *numpy_transitions,
        foobar=42,
    )
    zero_reward_net.predict_processed.assert_called_once_with(
        *numpy_transitions,
        foobar=42,
    )


def test_predict_processed_wrappers_to_pass_on_method_calls_to_base(
    numpy_transitions: NumpyTransitions,
    torch_transitions: TorchTransitions,
    predict_processed_wrapper: reward_nets.PredictProcessedWrapper,
):
    base = mock.create_autospec(testing_reward_nets.MockRewardNet)

    base.device = th.device("cpu")

    base.dtype = th.float32

    predict_processed_wrapper._base = base
    # Check method calls
    for attr, call_with, return_value in [
        ("forward", torch_transitions, th.zeros(10)),
        ("predict_th", numpy_transitions, np.zeros(10)),
        ("predict", numpy_transitions, np.zeros(10)),
        ("predict_processed", numpy_transitions, np.zeros(10)),
        ("preprocess", numpy_transitions, torch_transitions),
    ]:
        setattr(base, attr, mock.MagicMock(return_value=return_value))
        getattr(predict_processed_wrapper, attr)(*call_with)
        getattr(base, attr).assert_called_once_with(*call_with)

    # Check property lookups
    assert predict_processed_wrapper.device is base.device

    assert predict_processed_wrapper.dtype is base.dtype


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
def test_training_regression(normalize_input_layer, rng):
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
        transitions = rollout.generate_transitions(
            random,
            venv,
            n_timesteps=100,
            rng=rng,
        )
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
