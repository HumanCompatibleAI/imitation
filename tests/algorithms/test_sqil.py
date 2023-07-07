"""Tests `imitation.algorithms.sqil`."""

import gym
import numpy as np
import stable_baselines3.common.buffers as buffers
import stable_baselines3.common.vec_env as vec_env
import stable_baselines3.dqn as dqn
from stable_baselines3 import ppo
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import base as algo_base
from imitation.algorithms import sqil
from imitation.data import rollout, wrappers
from imitation.testing import reward_improvement


def test_sqil_demonstration_buffer(rng):
    env = gym.make("CartPole-v1")
    venv = vec_env.DummyVecEnv([lambda: wrappers.RolloutInfoWrapper(env)])
    policy = "MlpPolicy"

    sampling_agent = dqn.DQN(
        env=env,
        policy=policy,
    )

    rollouts = rollout.rollout(
        sampling_agent.policy,
        venv,
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
    )
    demonstrations = rollout.flatten_trajectories(rollouts)

    model = sqil.SQIL(
        venv=venv,
        demonstrations=demonstrations,
        policy=policy,
    )

    assert isinstance(model.dqn.replay_buffer, sqil.SQILReplayBuffer)
    expert_buffer = model.dqn.replay_buffer.expert_buffer

    # Check that demonstrations are stored in the replay buffer correctly
    for i in range(len(demonstrations)):
        obs = expert_buffer.observations[i]
        act = expert_buffer.actions[i]
        next_obs = expert_buffer.next_observations[i]
        done = expert_buffer.dones[i]

        np.testing.assert_array_equal(obs[0], demonstrations.obs[i])
        np.testing.assert_array_equal(act[0], demonstrations.acts[i])
        np.testing.assert_array_equal(next_obs[0], demonstrations.next_obs[i])
        np.testing.assert_array_equal(done, demonstrations.dones[i])


def test_sqil_demonstration_without_flatten(rng):
    env = gym.make("CartPole-v1")
    venv = vec_env.DummyVecEnv([lambda: wrappers.RolloutInfoWrapper(env)])
    policy = "MlpPolicy"

    sampling_agent = dqn.DQN(
        env=env,
        policy=policy,
    )

    rollouts = rollout.rollout(
        sampling_agent.policy,
        venv,
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
    )

    flat_rollouts = rollout.flatten_trajectories(rollouts)
    n_samples = len(flat_rollouts)

    model = sqil.SQIL(
        venv=venv,
        demonstrations=rollouts,
        policy=policy,
    )

    assert isinstance(model.dqn.replay_buffer, sqil.SQILReplayBuffer)
    assert isinstance(model.dqn.replay_buffer.expert_buffer, buffers.ReplayBuffer)

    assert len(model.dqn.replay_buffer.expert_buffer.observations) == n_samples


def test_sqil_demonstration_data_loader(rng):
    env = gym.make("CartPole-v1")
    venv = vec_env.DummyVecEnv([lambda: wrappers.RolloutInfoWrapper(env)])
    policy = "MlpPolicy"

    sampling_agent = dqn.DQN(
        env=env,
        policy=policy,
    )

    rollouts = rollout.rollout(
        sampling_agent.policy,
        venv,
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
    )

    transition_mappings = algo_base.make_data_loader(rollouts, batch_size=4)

    model = sqil.SQIL(
        venv=venv,
        demonstrations=transition_mappings,
        policy=policy,
    )

    assert isinstance(model.dqn.replay_buffer, sqil.SQILReplayBuffer)
    assert isinstance(model.dqn.replay_buffer.expert_buffer, buffers.ReplayBuffer)

    assert len(model.dqn.replay_buffer.expert_buffer.observations) == sum(
        len(traj["obs"]) for traj in transition_mappings
    )


def test_sqil_cartpole_no_crash(rng):
    env = gym.make("CartPole-v1")
    venv = vec_env.DummyVecEnv([lambda: wrappers.RolloutInfoWrapper(env)])

    policy = "MlpPolicy"
    sampling_agent = dqn.DQN(
        env=env,
        policy=policy,
    )

    rollouts = rollout.rollout(
        sampling_agent.policy,
        venv,
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
    )
    demonstrations = rollout.flatten_trajectories(rollouts)
    model = sqil.SQIL(
        venv=venv,
        demonstrations=demonstrations,
        policy=policy,
        dqn_kwargs=dict(learning_starts=1000),
    )
    model.train(total_timesteps=10_000)


def test_sqil_cartpole_few_demonstrations(rng):
    env = gym.make("CartPole-v1")
    venv = vec_env.DummyVecEnv([lambda: wrappers.RolloutInfoWrapper(env)])

    policy = "MlpPolicy"
    sampling_agent = dqn.DQN(
        env=env,
        policy=policy,
    )

    rollouts = rollout.rollout(
        sampling_agent.policy,
        venv,
        rollout.make_sample_until(min_timesteps=None, min_episodes=1),
        rng=rng,
    )

    demonstrations = rollout.flatten_trajectories(rollouts)
    demonstrations = demonstrations[:5]

    model = sqil.SQIL(
        venv=venv,
        demonstrations=demonstrations,
        policy=policy,
        dqn_kwargs=dict(learning_starts=10),
    )
    model.train(total_timesteps=1_000)


def test_sqil_performance(rng):
    env = gym.make("CartPole-v1")
    venv = vec_env.DummyVecEnv([lambda: wrappers.RolloutInfoWrapper(env)])

    expert = ppo.PPO(
        policy=ppo.MlpPolicy,
        env=env,
        seed=0,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0003,
        n_epochs=10,
        n_steps=64,
    )
    expert.learn(10_000)

    expert_reward, _ = evaluate_policy(expert, env, 10)
    print(expert_reward)

    rollouts = rollout.rollout(
        expert.policy,
        venv,
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
    )

    demonstrations = rollout.flatten_trajectories(rollouts)
    demonstrations = demonstrations[:5]

    model = sqil.SQIL(
        venv=venv,
        demonstrations=demonstrations,
        policy="MlpPolicy",
        dqn_kwargs=dict(learning_starts=1000),
    )

    rewards_before, _ = evaluate_policy(
        model.policy,
        env,
        10,
        return_episode_rewards=True,
    )

    model.train(total_timesteps=10_000)

    rewards_after, _ = evaluate_policy(
        model.policy,
        env,
        10,
        return_episode_rewards=True,
    )

    assert reward_improvement.is_significant_reward_improvement(
        rewards_before,
        rewards_after,
    )
