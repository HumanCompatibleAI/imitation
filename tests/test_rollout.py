"""Tests for code that generates trajectory rollouts."""

import functools
from typing import Mapping, Sequence

import gym
import numpy as np
import pytest
from stable_baselines3.common import monitor, vec_env

from imitation.data import rollout, types, wrappers
from imitation.policies import serialize
from imitation.policies.base import RandomPolicy


class TerminalSentinelEnv(gym.Env):
    def __init__(self, max_acts):
        self.max_acts = max_acts
        self.current_step = 0
        self.action_space = gym.spaces.Discrete(1)
        self.observation_space = gym.spaces.Box(np.array([0]), np.array([1]))

    def reset(self):
        self.current_step = 0
        return np.array([0])

    def step(self, action):
        self.current_step += 1
        done = self.current_step >= self.max_acts
        observation = np.array([1 if done else 0])
        rew = 0.0
        return observation, rew, done, {}


def _sample_fixed_length_trajectories(
    episode_lengths: Sequence[int],
    min_episodes: int,
    **kwargs,
) -> Sequence[types.Trajectory]:
    venv = vec_env.DummyVecEnv(
        [functools.partial(TerminalSentinelEnv, length) for length in episode_lengths]
    )
    policy = RandomPolicy(venv.observation_space, venv.action_space)
    sample_until = rollout.min_episodes(min_episodes)
    trajectories = rollout.generate_trajectories(
        policy,
        venv,
        sample_until=sample_until,
        **kwargs,
    )
    return trajectories


def test_complete_trajectories():
    """Checks trajectories include the terminal observation.

    This is hidden by default by VecEnv's auto-reset; we add it back in using
    `rollout.RolloutInfoWrapper`.
    """
    min_episodes = 13
    max_acts = 5
    num_envs = 4
    trajectories = _sample_fixed_length_trajectories(
        [max_acts] * num_envs, min_episodes
    )
    assert len(trajectories) >= min_episodes
    expected_obs = np.array([[0]] * max_acts + [[1]])
    for trajectory in trajectories:
        obs = trajectory.obs
        acts = trajectory.acts
        assert len(obs) == len(acts) + 1
        assert np.all(obs == expected_obs)


@pytest.mark.parametrize(
    "episode_lengths,min_episodes,expected_counts",
    [
        # Do we keep on sampling from the 1st (len 3) environment that remains 'alive'?
        ([3, 5], 2, {3: 2, 5: 1}),
        # Do we keep on sampling from the 2nd (len 7) environment that remains 'alive'?
        ([3, 7], 2, {3: 2, 7: 1}),
        # Similar, but more extreme case with num environments > num episodes
        ([3, 3, 3, 7], 2, {3: 3, 7: 1}),
        # Do we stop sampling at 2 episodes if we get two equal-length episodes?
        ([3, 3], 2, {3: 2}),
        ([5, 5], 2, {5: 2}),
        ([7, 7], 2, {7: 2}),
    ],
)
def test_unbiased_trajectories(
    episode_lengths: Sequence[int],
    min_episodes: int,
    expected_counts: Mapping[int, int],
):
    """Checks trajectories are sampled without bias towards shorter episodes.

    Specifically, we create a VecEnv consisting of environments with fixed-length
    `episode_lengths`. This is unrealistic and breaks the i.i.d. assumption, but lets
    us test things deterministically.

    If we hit `min_episodes` exactly and all environments are done at the same time,
    we should stop and not sample any more trajectories. Otherwise, we should keep
    sampling from any in-flight environments, but not add trajectories from any other
    environments.

    The different test cases check each of these cases.
    """
    trajectories = _sample_fixed_length_trajectories(episode_lengths, min_episodes)
    assert len(trajectories) == sum(expected_counts.values())
    traj_lens = np.array([len(traj) for traj in trajectories])
    for length, count in expected_counts.items():
        assert np.sum(traj_lens == length) == count


def test_seed_trajectories():
    """Check trajectory order deterministic given seed and that seed is not no-op.

    Note in general environments and policies are stochastic, so the trajectory
    order *will* differ unless environment/policy seeds are also set.

    However, `TerminalSentinelEnv` is fixed-length deterministic, so there are no
    such confounders in this test.
    """
    rng_a1 = np.random.RandomState(0)
    rng_a2 = np.random.RandomState(0)
    rng_b = np.random.RandomState(1)
    traj_a1 = _sample_fixed_length_trajectories([3, 5], 2, rng=rng_a1)
    traj_a2 = _sample_fixed_length_trajectories([3, 5], 2, rng=rng_a2)
    traj_b = _sample_fixed_length_trajectories([3, 5], 2, rng=rng_b)
    assert [len(traj) for traj in traj_a1] == [len(traj) for traj in traj_a2]
    assert [len(traj) for traj in traj_a1] != [len(traj) for traj in traj_b]


class ObsRewHalveWrapper(gym.Wrapper):
    """Simple wrapper that scales every reward and observation feature by 0.5."""

    def reset(self, **kwargs):
        obs = self.env.reset(**kwargs) / 2
        return obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        return obs / 2, rew / 2, done, info


def test_rollout_stats():
    """Applying `ObsRewIncrementWrapper` halves the reward mean.

    `rollout_stats` should reflect this.
    """
    env = gym.make("CartPole-v1")
    env = monitor.Monitor(env, None)
    env = ObsRewHalveWrapper(env)
    venv = vec_env.DummyVecEnv([lambda: env])

    policy = serialize.load_policy("zero", "UNUSED", venv)
    trajs = rollout.generate_trajectories(policy, venv, rollout.min_episodes(10))
    s = rollout.rollout_stats(trajs)

    np.testing.assert_allclose(s["return_mean"], s["monitor_return_mean"] / 2)
    np.testing.assert_allclose(s["return_std"], s["monitor_return_std"] / 2)
    np.testing.assert_allclose(s["return_min"], s["monitor_return_min"] / 2)
    np.testing.assert_allclose(s["return_max"], s["monitor_return_max"] / 2)


def test_unwrap_traj():
    """Check that unwrap_traj reverses `ObsRewIncrementWrapper`.

    Also check that unwrapping twice is a no-op.
    """
    env = gym.make("CartPole-v1")
    env = wrappers.RolloutInfoWrapper(env)
    env = ObsRewHalveWrapper(env)
    venv = vec_env.DummyVecEnv([lambda: env])

    policy = serialize.load_policy("zero", "UNUSED", venv)
    trajs = rollout.generate_trajectories(policy, venv, rollout.min_episodes(10))
    trajs_unwrapped = [rollout.unwrap_traj(t) for t in trajs]
    trajs_unwrapped_twice = [rollout.unwrap_traj(t) for t in trajs_unwrapped]

    for t, t_unwrapped in zip(trajs, trajs_unwrapped):
        np.testing.assert_allclose(t.acts, t_unwrapped.acts)
        np.testing.assert_allclose(t.obs, t_unwrapped.obs / 2)
        np.testing.assert_allclose(t.rews, t_unwrapped.rews / 2)

    for t1, t2 in zip(trajs_unwrapped, trajs_unwrapped_twice):
        np.testing.assert_equal(t1.acts, t2.acts)
        np.testing.assert_equal(t1.obs, t2.obs)
        np.testing.assert_equal(t1.rews, t2.rews)
