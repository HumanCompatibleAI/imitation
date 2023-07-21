"""Tests `imitation.algorithms.sqil`."""

import numpy as np
import pytest
from stable_baselines3.common import policies, vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import sqil
from imitation.testing import expert_trajectories, reward_improvement

EXPERT_DATA_TYPES = ["trajectories", "transitions"]


def get_demos(rng: np.random.Generator, pytestconfig: pytest.Config, data_type):
    cache = pytestconfig.cache
    assert cache is not None
    return expert_trajectories.make_expert_transition_loader(
        cache_dir=cache.mkdir("experts"),
        batch_size=4,
        expert_data_type=data_type,
        env_name="seals/CartPole-v0",
        rng=rng,
        num_trajectories=60,
        shuffle=False,
    )


@pytest.mark.parametrize("expert_data_type", EXPERT_DATA_TYPES)
def test_sqil_demonstration_buffer(
    rng: np.random.Generator,
    pytestconfig: pytest.Config,
    cartpole_venv: vec_env.VecEnv,
    expert_data_type: str,
):
    policy = "MlpPolicy"
    model = sqil.SQIL(
        venv=cartpole_venv,
        demonstrations=get_demos(rng, pytestconfig, expert_data_type),
        policy=policy,
    )

    assert isinstance(model.policy, policies.BasePolicy)
    assert isinstance(model.dqn.replay_buffer, sqil.SQILReplayBuffer)
    expert_buffer = model.dqn.replay_buffer.expert_buffer

    # Check that demonstrations are stored in the replay buffer correctly
    demonstrations = get_demos(rng, pytestconfig, "transitions")
    n_samples = len(demonstrations)
    assert len(model.dqn.replay_buffer.expert_buffer.observations) == n_samples
    for i in range(n_samples):
        obs = expert_buffer.observations[i]
        act = expert_buffer.actions[i]
        assert expert_buffer.next_observations is not None
        next_obs = expert_buffer.next_observations[i]
        done = expert_buffer.dones[i]

        np.testing.assert_array_equal(obs[0], demonstrations.obs[i])
        np.testing.assert_array_equal(act[0], demonstrations.acts[i])
        np.testing.assert_array_equal(next_obs[0], demonstrations.next_obs[i])
        np.testing.assert_array_equal(done, demonstrations.dones[i])


def test_sqil_cartpole_no_crash(
    rng: np.random.Generator,
    pytestconfig: pytest.Config,
    cartpole_venv: vec_env.VecEnv,
):
    policy = "MlpPolicy"
    model = sqil.SQIL(
        venv=cartpole_venv,
        demonstrations=get_demos(rng, pytestconfig, "transitions"),
        policy=policy,
        dqn_kwargs=dict(learning_starts=1000),
    )
    model.train(total_timesteps=10_000)


def test_sqil_cartpole_few_demonstrations(
    rng: np.random.Generator,
    pytestconfig: pytest.Config,
    cartpole_venv: vec_env.VecEnv,
):
    demonstrations = get_demos(rng, pytestconfig, "transitions")
    demonstrations = demonstrations[:5]

    policy = "MlpPolicy"
    model = sqil.SQIL(
        venv=cartpole_venv,
        demonstrations=demonstrations,
        policy=policy,
        dqn_kwargs=dict(learning_starts=10),
    )
    model.train(total_timesteps=1_000)


def test_sqil_performance(
    rng: np.random.Generator,
    pytestconfig: pytest.Config,
    cartpole_venv: vec_env.VecEnv,
):
    demonstrations = get_demos(rng, pytestconfig, "transitions")
    demonstrations = demonstrations[:20]

    model = sqil.SQIL(
        venv=cartpole_venv,
        demonstrations=demonstrations,
        policy="MlpPolicy",
        dqn_kwargs=dict(learning_starts=1000),
    )

    rewards_before, _ = evaluate_policy(
        model.policy,
        cartpole_venv,
        20,
        return_episode_rewards=True,
    )

    model.train(total_timesteps=10_000)

    rewards_after, _ = evaluate_policy(
        model.policy,
        cartpole_venv,
        20,
        return_episode_rewards=True,
    )

    assert reward_improvement.is_significant_reward_improvement(
        rewards_before,  # type:ignore[arg-type]
        rewards_after,  # type:ignore[arg-type]
    )
