import logging
import numbers
import os

import gym
import numpy as np
import pytest
import torch as th

from imitation.data import rollout, types
from imitation.policies import base
from imitation.rewards import reward_net, serialize
from imitation.util import util

ENVS = ["FrozenLake-v0", "CartPole-v1", "Pendulum-v0"]
HARDCODED_TYPES = ["zero"]

REWARD_NETS = [reward_net.BasicRewardNet, reward_net.BasicShapedRewardNet]


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


def _make_feed_dict(reward_net: reward_net.RewardNet, transitions: types.Transitions):
    return {
        reward_net.obs_ph: transitions.obs,
        reward_net.act_ph: transitions.acts,
        reward_net.next_obs_ph: transitions.next_obs,
        reward_net.done_ph: transitions.dones,
    }


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
        rewards["train"].append(net.reward_train(*trans_args))
        rewards["test"].append(net.reward_test(*trans_args))

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
