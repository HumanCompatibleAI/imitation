"""Tests `imitation.util.reward_wrapper`."""

import numpy as np
import pytest
import stable_baselines3 as sb3
from stable_baselines3.sac import policies as sac_policies

from imitation.data import rollout
from imitation.policies.base import RandomPolicy
from imitation.rewards import reward_nets, reward_wrapper
from imitation.util import networks, util


class FunkyReward:
    """A reward that ignores observation and depends only on batch index."""

    def __call__(self, obs, act, next_obs, steps=None):
        """Returns consecutive reward from 1 to batch size `len(obs)`."""
        # give each environment number from 1 to num_envs
        return (np.arange(len(obs)) + 1).astype("float32")


def test_reward_overwrite():
    """Test that reward wrapper actually overwrites base rewards."""
    env_name = "Pendulum-v1"
    num_envs = 3
    env = util.make_vec_env(env_name, num_envs)
    reward_fn = FunkyReward()
    wrapped_env = reward_wrapper.RewardVecEnvWrapper(env, reward_fn)
    policy = RandomPolicy(env.observation_space, env.action_space)
    sample_until = rollout.make_min_episodes(10)
    default_stats = rollout.rollout_stats(
        rollout.generate_trajectories(policy, env, sample_until),
    )
    wrapped_stats = rollout.rollout_stats(
        rollout.generate_trajectories(policy, wrapped_env, sample_until),
    )
    # Pendulum-v1 always has negative rewards
    assert default_stats["return_max"] < 0
    # ours gives between 1 * traj_len and num_envs * traj_len reward
    # (trajectories are all constant length of 200 in Pendulum)
    steps = wrapped_stats["len_mean"]
    assert wrapped_stats["return_min"] == 1 * steps
    assert wrapped_stats["return_max"] == num_envs * steps

    # check that wrapped reward is negative (all pendulum rewards is negative)
    # and other rewards are non-negative
    rand_act, _ = policy.predict(wrapped_env.reset())
    _, rew, _, infos = wrapped_env.step(rand_act)
    assert np.all(rew >= 0)
    assert np.all([info_dict["original_env_rew"] < 0 for info_dict in infos])


def reward_fn_ones(path, venv):
    del path, venv

    def f(
        obs: np.ndarray,
        act: np.ndarray,
        next_obs: np.ndarray,
        dones: np.ndarray,
    ) -> np.ndarray:
        del act, next_obs, dones  # Unused.
        return np.ones((obs.shape[0]))

    return f


def _check_zeros(arr: np.ndarray) -> bool:
    return (arr == 0.0).all()


def _check_ones(arr: np.ndarray) -> bool:
    return (arr == 1.0).all()


def _check_gt_rews(arr: np.ndarray) -> bool:
    return (arr != 0.0).all() and (arr != 1.0).all()


@pytest.mark.parametrize("num_envs", (4, 1))
@pytest.mark.parametrize("buffer_size", (1000, 450))
def test_reward_relabel_callback(buffer_size, num_envs):
    """Test that RewardRelabelCallback actually relabeled reward in replay buffer."""
    env_name = "Pendulum-v1"
    learn_steps = 100
    subvenv_steps = learn_steps // num_envs
    venv = util.make_vec_env(env_name, num_envs, seed=42)
    reward_fn = reward_fn_ones("foo", venv)
    rl_algo = sb3.SAC(
        policy=sac_policies.SACPolicy,
        policy_kwargs=dict(),
        env=venv,
        seed=42,
        learning_starts=0,
        buffer_size=buffer_size,
    )

    buffer = rl_algo.replay_buffer
    buffer_len = buffer.rewards.shape[0]
    pos_i = int(buffer.pos)
    assert (buffer.rewards == 0.0).all() and pos_i == 0

    # First iteration: no relabeling
    rl_algo.learn(total_timesteps=learn_steps)
    pos_j = int(buffer.pos)
    assert pos_j == subvenv_steps % buffer_len
    assert (buffer.rewards[:pos_j] != 0.0).any()  # seeded with 42

    # A few more iterations with relabeling
    reward_relabel_callback = reward_wrapper.RewardRelabelCallback(reward_fn=reward_fn)
    for i in range(2, 8):
        pos_i = int(pos_j)
        rl_algo.learn(total_timesteps=learn_steps, callback=reward_relabel_callback)
        pos_j = int(buffer.pos)
        assert pos_j == subvenv_steps * i % buffer_len
        if buffer.full:
            # check the case where the buffer is full
            assert pos_i != pos_j
            if pos_i < pos_j:
                assert _check_ones(buffer.rewards[:pos_i])
                assert _check_gt_rews(buffer.rewards[pos_i:pos_j])
                assert _check_ones(buffer.rewards[pos_j:])
            else:
                assert _check_gt_rews(buffer.rewards[pos_i:])
                assert _check_ones(buffer.rewards[pos_j:pos_i])
                assert _check_gt_rews(buffer.rewards[:pos_j])
        else:
            assert pos_i < pos_j
            assert _check_ones(buffer.rewards[:pos_i])
            assert _check_gt_rews(buffer.rewards[pos_i:pos_j])
            assert _check_zeros(buffer.rewards[pos_j:])


@pytest.mark.parametrize("normalize_reward", (True, False))
def test_reward_relabel_norm_reward(normalize_reward):
    """Test RewardRelabelCallback doesn't change norm stats in NormalizedRewardNet."""
    env_name = "Pendulum-v1"
    num_envs, learn_steps = 2, 100
    subvenv_steps = learn_steps // num_envs
    venv = util.make_vec_env(env_name, num_envs, seed=42)
    reward_net = base_reward_net = reward_nets.BasicRewardNet(
        venv.observation_space,
        venv.action_space,
        normalize_input_layer=networks.RunningNorm,
    )
    if normalize_reward:
        reward_net = reward_nets.NormalizedRewardNet(
            reward_net,
            normalize_output_layer=networks.RunningNorm,
        )
    reward_fn = reward_net.predict_processed

    # Create a reward relabel callback
    reward_relabel_callback = reward_wrapper.create_rew_relabel_callback(reward_fn)
    rl_algo = sb3.SAC(
        policy=sac_policies.SACPolicy,
        policy_kwargs=dict(),
        env=venv,
        seed=42,
        learning_starts=0,
        buffer_size=1000,
    )

    buffer = rl_algo.replay_buffer
    buffer_len = buffer.rewards.shape[0]
    pos_i = int(buffer.pos)
    assert (buffer.rewards == 0.0).all() and pos_i == 0

    # First iteration: no relabeling
    rl_algo.learn(total_timesteps=learn_steps)
    pos_j = int(buffer.pos)
    assert pos_j == subvenv_steps % buffer_len
    assert (buffer.rewards[:pos_j] != 0.0).any()  # seeded with 42

    # Four more iterations: relabeling
    for _ in range(2, 4):
        rl_algo.learn(total_timesteps=learn_steps, callback=reward_relabel_callback)
        assert base_reward_net.mlp.normalize_input.count.item() == 0
        if normalize_reward:
            assert reward_net.normalize_output_layer.count.item() == 0
