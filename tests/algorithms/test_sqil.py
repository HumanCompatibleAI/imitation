"""Tests `imitation.algorithms.sqil`."""
from typing import Any, Dict, Optional, Type
from unittest import mock

import numpy as np
import pytest
from stable_baselines3 import ddpg, dqn, sac, td3
from stable_baselines3.common import off_policy_algorithm, policies, vec_env
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import sqil
from imitation.testing import expert_trajectories, reward_improvement

EXPERT_DATA_TYPES = ["trajectories", "transitions"]
RL_ALGOS_CONT_ACTIONS = [ddpg.DDPG, sac.SAC, td3.TD3]


def get_demos(
    env_name: str,
    rng: np.random.Generator,
    pytestconfig: pytest.Config,
    data_type: str,
):
    cache = pytestconfig.cache
    assert cache is not None
    return expert_trajectories.make_expert_transition_loader(
        cache_dir=cache.mkdir("experts"),
        batch_size=4,
        expert_data_type=data_type,
        env_name=env_name,
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
        demonstrations=get_demos(
            "seals/CartPole-v0",
            rng,
            pytestconfig,
            expert_data_type,
        ),
        policy=policy,
    )

    assert isinstance(model.policy, policies.BasePolicy)
    assert isinstance(model.rl_algo.replay_buffer, sqil.SQILReplayBuffer)
    expert_buffer = model.rl_algo.replay_buffer.expert_buffer

    # Check that demonstrations are stored in the replay buffer correctly
    demonstrations = get_demos("seals/CartPole-v0", rng, pytestconfig, "transitions")
    n_samples = len(demonstrations)
    assert len(model.rl_algo.replay_buffer.expert_buffer.observations) == n_samples
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


def _test_sqil_no_crash(
    rng: np.random.Generator,
    pytestconfig: pytest.Config,
    venv: vec_env.VecEnv,
    env_name: str,
    rl_algo_class: Type[off_policy_algorithm.OffPolicyAlgorithm],
    rl_kwargs: Optional[Dict[str, Any]] = None,
):
    policy = "MlpPolicy"
    model = sqil.SQIL(
        venv=venv,
        demonstrations=get_demos(env_name, rng, pytestconfig, "transitions"),
        policy=policy,
        rl_algo_class=rl_algo_class,
        rl_kwargs=rl_kwargs,
    )
    model.train(total_timesteps=5000)


def test_sqil_no_crash_discrete(
    rng: np.random.Generator,
    pytestconfig: pytest.Config,
    cartpole_venv: vec_env.VecEnv,
):
    _test_sqil_no_crash(
        rng,
        pytestconfig,
        cartpole_venv,
        "seals/CartPole-v0",
        rl_algo_class=dqn.DQN,
        rl_kwargs=dict(learning_starts=1000),
    )


@pytest.mark.parametrize("rl_algo_class", RL_ALGOS_CONT_ACTIONS)
def test_sqil_no_crash_continuous(
    rng: np.random.Generator,
    pytestconfig: pytest.Config,
    pendulum_single_venv: vec_env.VecEnv,
    rl_algo_class: Type[off_policy_algorithm.OffPolicyAlgorithm],
):
    _test_sqil_no_crash(
        rng,
        pytestconfig,
        pendulum_single_venv,
        "Pendulum-v1",
        rl_algo_class=rl_algo_class,
    )


def _test_sqil_few_demonstrations(
    rng: np.random.Generator,
    pytestconfig: pytest.Config,
    venv: vec_env.VecEnv,
    env_name: str,
    rl_algo_class: Type[off_policy_algorithm.OffPolicyAlgorithm],
    rl_kwargs: Optional[Dict[str, Any]] = None,
):
    demonstrations = get_demos(env_name, rng, pytestconfig, "transitions")
    demonstrations = demonstrations[:5]

    policy = "MlpPolicy"
    model = sqil.SQIL(
        venv=venv,
        demonstrations=demonstrations,
        policy=policy,
        rl_algo_class=rl_algo_class,
        rl_kwargs=rl_kwargs,
    )
    model.train(total_timesteps=1_000)


def test_sqil_few_demonstrations_discrete(
    rng: np.random.Generator,
    pytestconfig: pytest.Config,
    cartpole_venv: vec_env.VecEnv,
):
    _test_sqil_few_demonstrations(
        rng,
        pytestconfig,
        cartpole_venv,
        "seals/CartPole-v0",
        rl_algo_class=dqn.DQN,
        rl_kwargs=dict(learning_starts=10),
    )


@pytest.mark.parametrize("rl_algo_class", RL_ALGOS_CONT_ACTIONS)
def test_sqil_few_demonstrations_continuous(
    rng: np.random.Generator,
    pytestconfig: pytest.Config,
    pendulum_single_venv: vec_env.VecEnv,
    rl_algo_class: Type[off_policy_algorithm.OffPolicyAlgorithm],
):
    _test_sqil_few_demonstrations(
        rng,
        pytestconfig,
        pendulum_single_venv,
        "Pendulum-v1",
        rl_algo_class=rl_algo_class,
    )


def _test_sqil_performance(
    rng: np.random.Generator,
    pytestconfig: pytest.Config,
    venv: vec_env.VecEnv,
    env_name: str,
    rl_algo_class: Type[off_policy_algorithm.OffPolicyAlgorithm],
    rl_kwargs: Optional[Dict[str, Any]] = None,
):
    SEED = 42
    demonstrations = get_demos(env_name, rng, pytestconfig, "transitions")
    model = sqil.SQIL(
        venv=venv,
        demonstrations=demonstrations,
        policy="MlpPolicy",
        rl_algo_class=rl_algo_class,
        rl_kwargs=rl_kwargs,
    )

    venv.seed(SEED)
    rewards_before, _ = evaluate_policy(
        model.policy,
        venv,
        100,
        return_episode_rewards=True,
    )

    model.train(total_timesteps=10_000)

    venv.seed(SEED)
    rewards_after, _ = evaluate_policy(
        model.policy,
        venv,
        100,
        return_episode_rewards=True,
    )

    assert reward_improvement.is_significant_reward_improvement(
        rewards_before,  # type:ignore[arg-type]
        rewards_after,  # type:ignore[arg-type]
    )


def test_sqil_performance_discrete(
    rng: np.random.Generator,
    pytestconfig: pytest.Config,
    cartpole_venv: vec_env.VecEnv,
):
    _test_sqil_performance(
        rng,
        pytestconfig,
        cartpole_venv,
        "seals/CartPole-v0",
        rl_algo_class=dqn.DQN,
        rl_kwargs=dict(
            learning_starts=500,
            learning_rate=0.002,
            batch_size=220,
            seed=42,
        ),
    )


@pytest.mark.parametrize("rl_algo_class", RL_ALGOS_CONT_ACTIONS)
def test_sqil_performance_continuous(
    rng: np.random.Generator,
    pytestconfig: pytest.Config,
    pendulum_single_venv: vec_env.VecEnv,
    rl_algo_class: Type[off_policy_algorithm.OffPolicyAlgorithm],
):
    _test_sqil_performance(
        rng,
        pytestconfig,
        pendulum_single_venv,
        "Pendulum-v1",
        rl_algo_class=rl_algo_class,
        rl_kwargs=dict(
            learning_starts=100,
            learning_rate=0.001,
            gamma=0.99,
            seed=42,
        ),
    )


@pytest.mark.parametrize("illegal_kw", ["replay_buffer_class", "replay_buffer_kwargs"])
def test_sqil_constructor_raises(illegal_kw: str):
    with pytest.raises(ValueError, match=".*SQIL uses a custom replay buffer.*"):
        sqil.SQIL(
            venv=mock.MagicMock(spec=vec_env.VecEnv),
            demonstrations=None,
            policy="MlpPolicy",
            rl_kwargs={illegal_kw: None},
        )
