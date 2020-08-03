import logging
import numbers
import os

import gym
import numpy as np
import pytest
import torch as th

from imitation.data import rollout
from imitation.policies import base
from imitation.rewards import reward_nets, serialize
from imitation.util import util

ENVS = ["FrozenLake-v0", "CartPole-v1", "Pendulum-v0"]
HARDCODED_TYPES = ["zero"]

REWARD_NETS = [reward_nets.BasicRewardNet, reward_nets.BasicShapedRewardNet]


@pytest.mark.parametrize("env_name", ENVS)
@pytest.mark.parametrize("reward_net_cls", REWARD_NETS)
def test_init_no_crash(env_name, reward_net_cls):
    env = gym.make(env_name)
    for i in range(3):
        reward_net_cls(env.observation_space, env.action_space)


def _sample(space, n):
    return np.array([space.sample() for _ in range(n)])


@pytest.mark.parametrize("env_name", ENVS)
@pytest.mark.parametrize("reward_type", HARDCODED_TYPES)
def test_reward_valid(env_name, reward_type):
    """Test output of reward function is appropriate shape and type."""
    venv = util.make_vec_env(env_name, n_envs=1, parallel=False)
    TRAJECTORY_LEN = 10
    obs = _sample(venv.observation_space, TRAJECTORY_LEN)
    actions = _sample(venv.action_space, TRAJECTORY_LEN)
    next_obs = _sample(venv.observation_space, TRAJECTORY_LEN)
    steps = np.arange(0, TRAJECTORY_LEN)

    reward_fn = serialize.load_reward(reward_type, "foobar", venv)
    pred_reward = reward_fn(obs, actions, next_obs, steps)

    assert pred_reward.shape == (TRAJECTORY_LEN,)
    assert isinstance(pred_reward[0], numbers.Number)


@pytest.mark.parametrize("env_name", ENVS)
@pytest.mark.parametrize("net_cls", REWARD_NETS)
def test_serialize_identity(env_name, net_cls, tmpdir):
    """Does output of deserialized reward network match that of original?"""
    logging.info(f"Testing {net_cls}")

    venv = util.make_vec_env(env_name, n_envs=1, parallel=False)
    original = net_cls(venv.observation_space, venv.action_space)
    random = base.RandomPolicy(venv.observation_space, venv.action_space)

    tmppath = os.path.join(tmpdir, "reward.pt")
    th.save(original, tmppath)
    loaded = th.load(tmppath)

    assert original.observation_space == loaded.observation_space
    assert original.action_space == loaded.action_space

    transitions = rollout.generate_transitions(random, venv, n_timesteps=100)

    unshaped_fn = serialize.load_reward("RewardNet_unshaped", tmppath, venv)
    shaped_fn = serialize.load_reward("RewardNet_shaped", tmppath, venv)
    rewards = {
        "train": [],
        "test": [],
    }
    for net in [original, loaded]:
        trans_args = (
            transitions.obs,
            transitions.acts,
            transitions.next_obs,
            transitions.dones,
        )
        rewards["train"].append(net.predict_reward_train(*trans_args))
        rewards["test"].append(net.predict_reward_test(*trans_args))

    args = (
        transitions.obs,
        transitions.acts,
        transitions.next_obs,
        transitions.dones,
    )
    rewards["train"].append(shaped_fn(*args))
    rewards["test"].append(unshaped_fn(*args))

    for key, predictions in rewards.items():
        assert len(predictions) == 3
        assert np.allclose(predictions[0], predictions[1])
        assert np.allclose(predictions[0], predictions[2])


class Env2D(gym.Env):
    """Mock environment with 2D observations."""

    def __init__(self):
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
    potential networks, leading to shape mismatches."""
    # instantiate environment & get batch observations, actions, etc.
    env = Env2D()
    obs = env.reset()
    action = env.action_space.sample()
    next_obs, _, done, _ = env.step(action)
    obs_b = obs[None]
    action_b = np.array([action], dtype="int")
    next_obs_b = next_obs[None]
    done_b = np.array([done], dtype="bool")

    net = reward_nets.BasicShapedRewardNet(env.observation_space, env.action_space)
    rew_batch = net.predict_reward_train(obs_b, action_b, next_obs_b, done_b)
    assert rew_batch.shape == (1,)
