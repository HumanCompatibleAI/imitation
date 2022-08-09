"""Test imitation.policies.replay_buffer_wrapper."""

import os.path as osp
from typing import Type

import numpy as np
import pytest
import stable_baselines3 as sb3
import torch as th
from stable_baselines3.common import buffers, off_policy_algorithm, policies
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.save_util import load_from_pkl

from imitation.policies.replay_buffer_wrapper import ReplayBufferRewardWrapper
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
) -> off_policy_algorithm.OffPolicyAlgorithm:
    venv = util.make_vec_env("Pendulum-v1", n_envs=1)
    rl_kwargs = dict(
        replay_buffer_class=ReplayBufferRewardWrapper,
        replay_buffer_kwargs=dict(
            replay_buffer_class=replay_buffer_class,
            reward_fn=zero_reward_fn,
        ),
    )
    rl_algo = rl_cls(
        policy=policy_cls,
        policy_kwargs=dict(),
        env=venv,
        seed=42,
        **rl_kwargs,
    )
    return rl_algo


def test_invalid_args():
    with pytest.raises(
        TypeError,
        match=r".*unexpected keyword argument 'replay_buffer_class'.*",
    ):
        make_algo_with_wrapped_buffer(
            rl_cls=sb3.PPO,
            policy_cls=policies.ActorCriticPolicy,
            replay_buffer_class=buffers.ReplayBuffer,
        )

    with pytest.raises(AssertionError, match=r".*only ReplayBuffer is supported.*"):
        make_algo_with_wrapped_buffer(
            rl_cls=sb3.SAC,
            policy_cls=sb3.sac.policies.SACPolicy,
            replay_buffer_class=buffers.DictReplayBuffer,
        )


def test_wrapper_class(tmpdir):
    rl_algo = make_algo_with_wrapped_buffer(
        rl_cls=sb3.SAC,
        policy_cls=sb3.sac.policies.SACPolicy,
        replay_buffer_class=buffers.ReplayBuffer,
    )

    total_timesteps = 20
    rl_algo.learn(total_timesteps=total_timesteps)

    buffer_path = osp.join(tmpdir, "buffer.pkl")
    rl_algo.save_replay_buffer(buffer_path)
    replay_buffer_wrapper = load_from_pkl(buffer_path)
    replay_buffer = replay_buffer_wrapper.replay_buffer

    # replay_buffer_wrapper.sample(...) should return zero-reward transitions
    assert total_timesteps == replay_buffer_wrapper.size() == replay_buffer.size()
    assert (replay_buffer_wrapper.sample(total_timesteps).rewards == 0.0).all()
    assert (replay_buffer.sample(total_timesteps).rewards != 0.0).all()  # seed=42

    # reset()
    replay_buffer_wrapper.reset()
    assert 0 == replay_buffer_wrapper.size() == replay_buffer.size()

    # to_torch()
    tensor = replay_buffer_wrapper.to_torch(np.ones(42))
    assert type(tensor) is th.Tensor

    # raise error for _get_samples()
    with pytest.raises(NotImplementedError, match=r".*_get_samples.*"):
        replay_buffer_wrapper._get_samples()
