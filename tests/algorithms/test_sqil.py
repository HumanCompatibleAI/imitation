import gym
import numpy as np
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.algorithms.sqil import SQIL
from imitation.data import rollout, types, wrappers


def test_sqil_demonstration_buffer(rng):
    env = gym.make("CartPole-v1")
    venv = DummyVecEnv([lambda: env])
    policy = "MlpPolicy"

    sampling_agent = SQIL(
        venv=venv,
        demonstrations=None,
        policy=policy,
    )

    rollouts = rollout.rollout(
        sampling_agent.policy,
        DummyVecEnv([lambda: wrappers.RolloutInfoWrapper(env)]),
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
    )
    demonstrations = rollout.flatten_trajectories(rollouts)

    model = SQIL(
        venv=venv,
        demonstrations=demonstrations,
        policy=policy,
    )

    # Check that demonstrations are stored in the replay buffer correctly
    for i in range(len(demonstrations)):
        obs = model.expert_buffer.observations[i]
        act = model.expert_buffer.actions[i]
        next_obs = model.expert_buffer.next_observations[i]
        done = model.expert_buffer.dones[i]

        np.testing.assert_array_equal(obs[0], demonstrations.obs[i])
        np.testing.assert_array_equal(act[0], demonstrations.acts[i])
        np.testing.assert_array_equal(next_obs[0], demonstrations.next_obs[i])
        np.testing.assert_array_equal(done, demonstrations.dones[i])


def test_sqil_cartpole_no_crash(rng):
    env = gym.make("CartPole-v1")
    venv = DummyVecEnv([lambda: env])

    policy = "MlpPolicy"
    sampling_agent = SQIL(
        venv=venv,
        demonstrations=None,
        policy=policy,
    )

    rollouts = rollout.rollout(
        sampling_agent.policy,
        DummyVecEnv([lambda: wrappers.RolloutInfoWrapper(env)]),
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
    )
    demonstrations = rollout.flatten_trajectories(rollouts)
    model = SQIL(
        venv=venv,
        demonstrations=demonstrations,
        policy=policy,
    )
    model.train(total_timesteps=100)
