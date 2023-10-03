"""Tests for `imitation.algorithms.density_baselines`."""

from dataclasses import asdict
from typing import Sequence, cast

import gymnasium as gym
import numpy as np
import pytest
import stable_baselines3
from stable_baselines3.common import envs as sb_envs
from stable_baselines3.common import policies, vec_env

from imitation.algorithms.density import DensityAlgorithm, DensityType
from imitation.data import rollout, types
from imitation.data.types import TrajectoryWithRew
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.policies.base import RandomPolicy
from imitation.testing import reward_improvement

parametrize_density_stationary = pytest.mark.parametrize(
    "density_type,is_stationary",
    [(density_type, True) for density_type in DensityType]
    + [(DensityType.STATE_DENSITY, False)],
)


def score_trajectories(
    trajectories: Sequence[types.Trajectory],
    density_reward: DensityAlgorithm,
):
    # score trajectories under given reward function w/o discount
    returns = []
    for traj in trajectories:
        dones = np.zeros(len(traj), dtype=bool)
        dones[-1] = True
        steps = np.arange(0, len(traj.acts))
        obs = types.assert_not_dictobs(traj.obs)
        rewards = density_reward(obs[:-1], traj.acts, obs[1:], dones, steps)
        ret = np.sum(rewards)
        returns.append(ret)
    return returns


# test on Pendulum rather than Cartpole because I don't handle episodes that
# terminate early yet (see issue #40)
@parametrize_density_stationary
def test_density_reward(
    density_type,
    is_stationary,
    pendulum_venv,
    pendulum_expert_trajectories: Sequence[TrajectoryWithRew],
    rng,
):
    # use only a subset of trajectories
    n_experts = len(pendulum_expert_trajectories)
    expert_trajectories_train = pendulum_expert_trajectories[: n_experts // 2]
    reward_fn = DensityAlgorithm(
        demonstrations=expert_trajectories_train,
        density_type=density_type,
        kernel="gaussian",
        venv=pendulum_venv,
        is_stationary=is_stationary,
        kernel_bandwidth=0.2,
        standardise_inputs=True,
        rng=rng,
    )
    reward_fn.train()

    # check that expert policy does better than a random policy under our reward
    # function
    random_policy = RandomPolicy(
        pendulum_venv.observation_space,
        pendulum_venv.action_space,
    )
    sample_until = rollout.make_min_episodes(15)
    random_trajectories = rollout.generate_trajectories(
        random_policy,
        pendulum_venv,
        sample_until=sample_until,
        rng=rng,
    )
    expert_trajectories_test = pendulum_expert_trajectories[n_experts // 2 :]
    random_returns = score_trajectories(random_trajectories, reward_fn)
    expert_returns = score_trajectories(expert_trajectories_test, reward_fn)
    assert reward_improvement.is_significant_reward_improvement(
        random_returns,
        expert_returns,
    )


@pytest.mark.expensive
def test_density_trainer_smoke(
    pendulum_venv,
    pendulum_expert_trajectories: Sequence[TrajectoryWithRew],
    rng,
):
    # tests whether density trainer runs, not whether it's good
    # (it's actually really poor)
    rollouts = pendulum_expert_trajectories[:2]
    rl_algo = stable_baselines3.PPO(policies.ActorCriticPolicy, pendulum_venv)
    density_trainer = DensityAlgorithm(
        demonstrations=rollouts,
        venv=pendulum_venv,
        rl_algo=rl_algo,
        rng=rng,
    )
    density_trainer.train()
    density_trainer.train_policy(n_timesteps=2)
    density_trainer.test_policy(n_trajectories=2)


def test_density_with_other_trajectory_types(
    pendulum_expert_trajectories: Sequence[TrajectoryWithRew],
    pendulum_venv,
    rng,
):
    rl_algo = stable_baselines3.PPO(
        policies.ActorCriticPolicy,
        pendulum_venv,
        n_steps=10,  # small value to make test faster
        n_epochs=2,  # small value to make test faster
    )
    rollouts = pendulum_expert_trajectories[:2]
    transitions = rollout.flatten_trajectories_with_rew(rollouts)
    transitions_mappings = [cast(types.TransitionMapping, asdict(transitions))]
    d = DensityAlgorithm(
        demonstrations=transitions_mappings,
        venv=pendulum_venv,
        rl_algo=rl_algo,
        rng=rng,
    )
    d.train()
    d.train_policy(n_timesteps=2)
    d.test_policy(n_trajectories=2)

    minimal_transitions = types.TransitionsMinimal(
        obs=transitions.obs,
        acts=transitions.acts,
        infos=transitions.infos,
    )
    d = DensityAlgorithm(
        demonstrations=minimal_transitions,
        venv=pendulum_venv,
        rl_algo=rl_algo,
        rng=rng,
    )
    d.train()
    d.train_policy(n_timesteps=2)
    d.test_policy(n_trajectories=2)


def test_density_trainer_raises(
    pendulum_venv,
    rng,
):
    rl_algo = stable_baselines3.PPO(policies.ActorCriticPolicy, pendulum_venv)
    density_trainer = DensityAlgorithm(
        venv=pendulum_venv,
        rl_algo=rl_algo,
        rng=rng,
        demonstrations=None,
        density_type=DensityType.STATE_STATE_DENSITY,
    )
    with pytest.raises(ValueError, match="STATE_STATE_DENSITY requires next_obs_b"):
        density_trainer._get_demo_from_batch(
            np.zeros((1, 3)),
            np.zeros((1, 1)),
            None,
        )

    with pytest.raises(TypeError, match="Unsupported demonstration type"):
        density_trainer.set_demonstrations("foo")  # type: ignore[arg-type]


# TODO(GH#794): Remove after https://github.com/DLR-RM/stable-baselines3/pull/1676
# merged and released.
class FloatReward(gym.RewardWrapper):
    """Typecasts reward to a float."""

    def reward(self, reward):
        return float(reward)


@parametrize_density_stationary
def test_dict_space(density_type, is_stationary):
    def make_env():
        env = sb_envs.SimpleMultiObsEnv(channel_last=False)
        env = FloatReward(env)
        return RolloutInfoWrapper(env)

    venv = vec_env.DummyVecEnv([make_env, make_env])

    # multi-input policy to accept dict observations
    rl_algo = stable_baselines3.PPO(policies.MultiInputActorCriticPolicy, venv)
    rng = np.random.default_rng()

    # sample random transitions
    rollouts = rollout.rollout(
        policy=None,
        venv=venv,
        sample_until=rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
        unwrap=True,
    )
    density_trainer = DensityAlgorithm(
        demonstrations=rollouts,
        density_type=density_type,
        kernel="gaussian",
        venv=venv,
        is_stationary=is_stationary,
        rl_algo=rl_algo,
        kernel_bandwidth=0.2,
        standardise_inputs=True,
        rng=rng,
    )
    # confirm that training works
    density_trainer.train()
