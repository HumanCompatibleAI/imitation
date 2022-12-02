"""Test imitation.policies.replay_buffer_wrapper."""

import os.path as osp
from typing import Type
from unittest.mock import Mock

import gym
import numpy as np
import pytest
import stable_baselines3 as sb3
import torch as th
from gym import spaces
from stable_baselines3.common import buffers, off_policy_algorithm, policies
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.preprocessing import get_action_dim, get_obs_shape
from stable_baselines3.common.save_util import load_from_pkl

from imitation.policies.replay_buffer_wrapper import (
    ReplayBufferAwareRewardFn,
    ReplayBufferRewardWrapper,
)
from imitation.util import util


def zero_reward_fn(
    state: np.ndarray,
    action: np.ndarray,
    next_state: np.ndarray,
    done: np.ndarray,
) -> np.ndarray:
    del action, next_state, done
    return np.zeros(state.shape[0], dtype=np.float32)


def make_algo_with_wrapped_buffer(
    rl_cls: Type[off_policy_algorithm.OffPolicyAlgorithm],
    policy_cls: Type[BasePolicy],
    replay_buffer_class: Type[buffers.ReplayBuffer],
    rng: np.random.Generator,
    buffer_size: int = 100,
) -> off_policy_algorithm.OffPolicyAlgorithm:
    venv = util.make_vec_env("Pendulum-v1", n_envs=1, rng=rng)
    rl_algo = rl_cls(
        policy=policy_cls,
        policy_kwargs=dict(),
        env=venv,
        seed=42,
        replay_buffer_class=ReplayBufferRewardWrapper,
        replay_buffer_kwargs=dict(
            replay_buffer_class=replay_buffer_class,
            reward_fn=zero_reward_fn,
        ),
        buffer_size=buffer_size,
    )  # type: ignore[call-arg]
    return rl_algo


def test_invalid_args(rng):
    with pytest.raises(
        TypeError,
        match=r".*unexpected keyword argument 'replay_buffer_class'.*",
    ):
        # we ignore the type because we are intentionally
        # passing the wrong type for the test
        make_algo_with_wrapped_buffer(
            rl_cls=sb3.PPO,  # type: ignore[arg-type]
            policy_cls=policies.ActorCriticPolicy,
            replay_buffer_class=buffers.ReplayBuffer,
            rng=rng,
        )

    with pytest.raises(AssertionError, match=r".*only ReplayBuffer is supported.*"):
        make_algo_with_wrapped_buffer(
            rl_cls=sb3.SAC,
            policy_cls=sb3.sac.policies.SACPolicy,
            replay_buffer_class=buffers.DictReplayBuffer,
            rng=rng,
        )


def test_wrapper_class(tmpdir, rng):
    buffer_size = 15
    total_timesteps = 20

    rl_algo = make_algo_with_wrapped_buffer(
        rl_cls=sb3.SAC,
        policy_cls=sb3.sac.policies.SACPolicy,
        replay_buffer_class=buffers.ReplayBuffer,
        buffer_size=buffer_size,
        rng=rng,
    )

    rl_algo.learn(total_timesteps=total_timesteps)

    buffer_path = osp.join(tmpdir, "buffer.pkl")
    rl_algo.save_replay_buffer(buffer_path)
    replay_buffer_wrapper = load_from_pkl(buffer_path)
    replay_buffer = replay_buffer_wrapper.replay_buffer

    # replay_buffer_wrapper.sample(...) should return zero-reward transitions
    assert buffer_size == replay_buffer_wrapper.size() == replay_buffer.size()
    assert (replay_buffer_wrapper.sample(total_timesteps).rewards == 0.0).all()
    assert (replay_buffer.sample(total_timesteps).rewards != 0.0).all()  # seed=42

    # replay_buffer_wrapper.pos, replay_buffer_wrapper.full
    assert replay_buffer_wrapper.pos == total_timesteps - buffer_size
    assert replay_buffer_wrapper.full

    # reset()
    replay_buffer_wrapper.reset()
    assert 0 == replay_buffer_wrapper.size() == replay_buffer.size()
    assert replay_buffer_wrapper.pos == 0
    assert not replay_buffer_wrapper.full

    # to_torch()
    tensor = replay_buffer_wrapper.to_torch(np.ones(42))
    assert type(tensor) is th.Tensor

    # raise error for _get_samples()
    with pytest.raises(NotImplementedError, match=r".*_get_samples.*"):
        replay_buffer_wrapper._get_samples()


class ActionIsObsEnv(gym.Env):
    """Simple environment where the obs is the action."""

    def __init__(self):
        """Initialize environment."""
        super().__init__()
        self.action_space = spaces.Box(np.array([0]), np.array([1]))
        self.observation_space = spaces.Box(np.array([0]), np.array([1]))

    def step(self, action):
        obs = action
        reward = 0
        done = False
        info = {}
        return obs, reward, done, info

    def reset(self):
        return np.array([0])


def test_replay_buffer_view_provides_buffered_observations():
    space = spaces.Box(np.array([0]), np.array([5]))
    n_envs = 2
    buffer_size = 10
    action = np.empty((n_envs, get_action_dim(space)))

    obs_shape = get_obs_shape(space)
    wrapper = ReplayBufferRewardWrapper(
        buffer_size,
        space,
        space,
        replay_buffer_class=ReplayBuffer,
        reward_fn=Mock(),
        n_envs=n_envs,
        handle_timeout_termination=False,
    )
    view = wrapper.buffer_view

    # initially empty
    assert len(view.observations) == 0

    # after adding observation
    obs1 = np.random.random((n_envs, *obs_shape))
    wrapper.add(obs1, obs1, action, np.empty(n_envs), np.empty(n_envs), [])
    np.testing.assert_allclose(view.observations, np.array([obs1]))

    # after filling buffer
    observations = np.random.random((buffer_size // n_envs, n_envs, *obs_shape))
    for obs in observations:
        wrapper.add(obs, obs, action, np.empty(n_envs), np.empty(n_envs), [])

    # ReplayBuffer internally uses a circular buffer
    expected = np.roll(observations, 1, axis=0)
    np.testing.assert_allclose(view.observations, expected)


def test_replay_buffer_reward_wrapper_calls_reward_initialization_callback():
    reward_fn = Mock(spec=ReplayBufferAwareRewardFn)
    buffer = ReplayBufferRewardWrapper(
        10,
        spaces.Discrete(2),
        spaces.Discrete(2),
        replay_buffer_class=ReplayBuffer,
        reward_fn=reward_fn,
        n_envs=2,
        handle_timeout_termination=False,
    )
    assert reward_fn.on_replay_buffer_initialized.call_args.args[0] is buffer
