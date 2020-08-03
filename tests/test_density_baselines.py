"""Testing simple density estimation baselines for IRL."""

from typing import Sequence

import numpy as np
import pytest

from imitation.algorithms.density_baselines import (
    STATE_ACTION_DENSITY,
    STATE_DENSITY,
    STATE_STATE_DENSITY,
    DensityReward,
    DensityTrainer,
)
from imitation.data import rollout, types
from imitation.policies.base import RandomPolicy
from imitation.rewards import common
from imitation.util import util

parametrize_density_stationary = pytest.mark.parametrize(
    "density_type,is_stationary",
    [
        (STATE_DENSITY, True),
        (STATE_DENSITY, False),
        (STATE_ACTION_DENSITY, True),
        (STATE_STATE_DENSITY, True),
    ],
)


def score_trajectories(
    trajectories: Sequence[types.Trajectory], reward_fn: common.RewardFn
):
    # score trajectories under given reward function w/o discount
    returns = []
    for traj in trajectories:
        steps = np.arange(0, len(traj.acts))
        rewards = reward_fn(traj.obs[:-1], traj.acts, traj.obs[1:], steps)
        ret = np.sum(rewards)
        returns.append(ret)
    return np.mean(returns)


@parametrize_density_stationary
def test_density_reward(density_type, is_stationary):
    # test on Pendulum rather than Cartpole because I don't handle episodes that
    # terminate early yet (see issue #40)
    env_name = "Pendulum-v0"
    env = util.make_vec_env(env_name, 2)

    # construct density-based reward from expert rollouts
    rollout_path = "tests/data/expert_models/pendulum_0/rollouts/final.pkl"
    # use only a subset of trajectories
    expert_trajectories_all = types.load(rollout_path)[:8]
    n_experts = len(expert_trajectories_all)
    expert_trajectories_train = expert_trajectories_all[: n_experts // 2]
    reward_fn = DensityReward(
        trajectories=expert_trajectories_train,
        density_type=density_type,
        kernel="gaussian",
        obs_space=env.observation_space,
        act_space=env.action_space,
        is_stationary=is_stationary,
        kernel_bandwidth=0.2,
        standardise_inputs=True,
    )

    # check that expert policy does better than a random policy under our reward
    # function
    random_policy = RandomPolicy(env.observation_space, env.action_space)
    sample_until = rollout.min_episodes(15)
    random_trajectories = rollout.generate_trajectories(
        random_policy, env, sample_until=sample_until
    )
    expert_trajectories_test = expert_trajectories_all[n_experts // 2 :]
    random_score = score_trajectories(random_trajectories, reward_fn)
    expert_score = score_trajectories(expert_trajectories_test, reward_fn)
    assert expert_score > random_score


@pytest.mark.expensive
def test_density_trainer_smoke():
    # tests whether density trainer runs, not whether it's good
    # (it's actually really poor)
    env_name = "Pendulum-v0"
    rollout_path = "tests/data/expert_models/pendulum_0/rollouts/final.pkl"
    rollouts = types.load(rollout_path)[:2]
    env = util.make_vec_env(env_name, 2)
    imitation_trainer = util.init_rl(env)
    density_trainer = DensityTrainer(
        env,
        rollouts=rollouts,
        imitation_trainer=imitation_trainer,
        density_type=STATE_ACTION_DENSITY,
        is_stationary=False,
        kernel="gaussian",
    )
    density_trainer.train_policy(n_timesteps=2)
    density_trainer.test_policy(n_trajectories=2)
