"""Tests `imitation.rewards.reward_nets` and `imitation.rewards.serialize`."""

import logging
import numbers
import os

import gym
import numpy as np
import pytest
import torch as th
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.data import rollout
from imitation.policies import base
from imitation.rewards import reward_nets, serialize
from imitation.util import networks, util

ENVS = ["FrozenLake-v1", "CartPole-v1", "Pendulum-v1"]
DESERIALIZATION_TYPES = [
    "zero",
    "RewardNet_normalized",
    "RewardNet_unnormalized",
    "RewardNet_shaped",
    "RewardNet_unshaped",
]

REWARD_NETS = [
    reward_nets.BasicRewardNet,
    reward_nets.BasicShapedRewardNet,
]
REWARD_NET_KWARGS = [
    {},
    {"normalize_input_layer": networks.RunningNorm},
]


@pytest.mark.parametrize("env_name", ENVS)
@pytest.mark.parametrize("reward_net_cls", REWARD_NETS)
@pytest.mark.parametrize("reward_net_kwargs", REWARD_NET_KWARGS)
@pytest.mark.parametrize("normalize_output_layer", [None, networks.RunningNorm])
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


def _potential(x):
    return th.zeros(1)


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
@pytest.mark.parametrize("net_cls", REWARD_NETS)
@pytest.mark.parametrize("normalize_rewards", [True, False])
def test_serialize_identity(env_name, net_cls, normalize_rewards, tmpdir):
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
    if isinstance(original, reward_nets.ShapedRewardNet):
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
            th.as_tensor(transitions.rews, device=norm_rew_net.device),
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
