"""Testing simple density estimation baselines for IRL."""

import numpy as np
import pytest

from imitation import util
from imitation.density_baselines import (STATE_ACTION_DENSITY, STATE_DENSITY,
                                         STATE_STATE_DENSITY, DensityReward,
                                         DensityTrainer)


def parametrize_density_stationary(fn):
  decorator = pytest.mark.parametrize("density_type,is_stationary",
                                      [(STATE_DENSITY, True),
                                       (STATE_DENSITY, False),
                                       (STATE_ACTION_DENSITY, True),
                                       (STATE_STATE_DENSITY, True)])
  return decorator(fn)


def score_trajectories(trajectories, reward_fn):
  # score trajectories under given reward function w/o discount
  returns = []
  for traj in trajectories:
    traj_zip = zip(traj['obs'], traj['act'], traj['obs'][1:])
    ret = 0.0
    for step, (obs, act, next_obs) in enumerate(traj_zip):
      ret += reward_fn([obs], [act], [next_obs], steps=[step])
    returns.append(ret)
  return np.mean(returns)


@parametrize_density_stationary
def test_density_reward(density_type, is_stationary):
  # use Pendulum b/c I don't handle termination correctly yet
  env_id = 'Pendulum-v0'
  env = util.make_vec_env(env_id, 2)

  # construct density-based reward from expert rollouts
  expert_trainer, = util.load_policy(env)
  n_episodes = 5
  expert_trajectories = util.generate_trajectories(expert_trainer,
                                                   env,
                                                   n_episodes=n_episodes)
  reward_fn = DensityReward(trajectories=expert_trajectories,
                            density_type=density_type,
                            kernel='gaussian',
                            obs_space=env.observation_space,
                            act_space=env.action_space,
                            is_stationary=is_stationary,
                            kernel_bandwidth=0.2,
                            standardise_inputs=True)

  # check that expert policy does better than a random policy under our reward
  # function
  random_policy = util.RandomPolicy(env.observation_space, env.action_space)
  random_trajectories = util.generate_trajectories(random_policy,
                                                   env,
                                                   n_episodes=n_episodes)
  unseen_expert_trajectories = util.generate_trajectories(expert_trainer,
                                                          env,
                                                          n_episodes=n_episodes)
  random_score = score_trajectories(random_trajectories, reward_fn)
  expert_score = score_trajectories(unseen_expert_trajectories, reward_fn)
  assert expert_score > random_score


@pytest.mark.expensive
@parametrize_density_stationary
def test_density_trainer(density_type, is_stationary):
  env_id = 'Pendulum-v0'
  env = util.make_vec_env(env_id, 2)
  expert_algo, = util.load_policy(env)
  imitation_trainer = util.make_blank_policy(env)
  density_trainer = DensityTrainer(env,
                                   expert_trainer=expert_algo,
                                   imitation_trainer=imitation_trainer,
                                   n_expert_trajectories=5,
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
