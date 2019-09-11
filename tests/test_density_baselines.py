"""Testing simple density estimation baselines for IRL."""

from typing import Sequence

import numpy as np
import pytest

from imitation import util
from imitation.algorithms.density_baselines import (STATE_ACTION_DENSITY,
                                                    STATE_DENSITY,
                                                    STATE_STATE_DENSITY,
                                                    DensityReward,
                                                    DensityTrainer)
from imitation.policies.base import RandomPolicy
from imitation.util import reward_wrapper, rollout

parametrize_density_stationary = pytest.mark.parametrize(
  "density_type,is_stationary",
  [(STATE_DENSITY, True),
   (STATE_DENSITY, False),
   (STATE_ACTION_DENSITY, True),
   (STATE_STATE_DENSITY, True)])


def score_trajectories(trajectories: Sequence[rollout.Trajectory],
                       reward_fn: reward_wrapper.RewardFn):
  # score trajectories under given reward function w/o discount
  returns = []
  for traj in trajectories:
    steps = np.arange(0, len(traj.act))
    rewards = reward_fn(traj.obs[:-1], traj.act, traj.obs[1:], steps)
    ret = np.sum(rewards)
    returns.append(ret)
  return np.mean(returns)


@parametrize_density_stationary
def test_density_reward(density_type, is_stationary):
  # test on Pendulum rather than Cartpole because I don't handle episodes that
  # terminate early yet (see issue #40)
  env_id = 'Pendulum-v0'
  env = util.make_vec_env(env_id, 2)

  # construct density-based reward from expert rollouts
  pattern = f"tests/data/rollouts/{env_id}_*.pkl"
  expert_trajectories_all = rollout.load_trajectories(pattern)
  n_experts = len(expert_trajectories_all)
  expert_trajectories_train = expert_trajectories_all[:n_experts // 2]
  reward_fn = DensityReward(trajectories=expert_trajectories_train,
                            density_type=density_type,
                            kernel='gaussian',
                            obs_space=env.observation_space,
                            act_space=env.action_space,
                            is_stationary=is_stationary,
                            kernel_bandwidth=0.2,
                            standardise_inputs=True)

  # check that expert policy does better than a random policy under our reward
  # function
  random_policy = RandomPolicy(env.observation_space, env.action_space)
  sample_until = rollout.min_episodes(n_experts // 2)
  random_trajectories = rollout.generate_trajectories(random_policy,
                                                      env,
                                                      sample_until=sample_until)
  expert_trajectories_test = expert_trajectories_all[n_experts // 2:]
  random_score = score_trajectories(random_trajectories, reward_fn)
  expert_score = score_trajectories(expert_trajectories_test, reward_fn)
  assert expert_score > random_score


@pytest.mark.expensive
@parametrize_density_stationary
def test_density_trainer(density_type, is_stationary):
  env_id = 'Pendulum-v0'
  pattern = f"tests/data/rollouts/{env_id}_*.pkl"
  rollouts = rollout.load_trajectories(pattern)
  env = util.make_vec_env(env_id, 2)
  imitation_trainer = util.init_rl(env)
  density_trainer = DensityTrainer(env,
                                   rollouts=rollouts,
                                   imitation_trainer=imitation_trainer,
                                   density_type=density_type,
                                   is_stationary=is_stationary,
                                   kernel='gaussian')
  novice_stats = density_trainer.test_policy()
  density_trainer.train_policy(2000)
  good_stats = density_trainer.test_policy()
  # Novice is bad
  assert novice_stats["return_mean"] < -500
  # Density is also pretty bad, but shouldn't make things more than 50% worse.
  # It would be nice to have a less flaky/more meaningful test here.
  assert good_stats["return_mean"] > 1.5 * novice_stats["return_mean"]
