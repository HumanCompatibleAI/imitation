"""
MVP-not-really
"""

import numpy as np
import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv
from stable_baselines import PPO1, PPO2


def _basics1(env_name="CartPole-v1", skip_training=True):
    """
    * Load X environment.
    * Train a PPO policy on frozen lake.
    * Generate 9 trajectory rollouts.
    * Print the state action pairs.
    BONUS:
    # Pickle the policy.
    # Load the policy.
    # Repeat the procedure above from the rollout part.
    """
    orig_env = gym.make(env_name)
    # # PPO1
    env = DummyVecEnv([lambda: orig_env])
    savepath = "saved_models/ppo_"+env_name
    if not skip_training:
        model = PPO1(MlpPolicy, env, verbose=1,
                tensorboard_log="./output/{}/".format(env_name))
        model.learn(50000)
        model.save(savepath)
    else:
        model = PPO1.load(savepath, env=env)

    X = generate_rollouts(model, orig_env, n_timesteps=3)
    print(X)


def generate_rollouts(model, env, n_timesteps):
    """
    Generate state-action pairs from a policy.

    Params:
    model -- A stable_baselines Model, trained on the gym environment.
    env (Env) -- A Gym Env. VecEnv is not currently supported.
    n_timesteps (int) -- The number of state-action pairs to collect.

    Return:
    rollout_obs (array) -- A numpy array with shape
      `[n_timesteps] + env.observation_space.shape`.
    rollout_act (array) -- A numpy array with shape
      `[n_timesteps] + env.action_space.shape`.
    """
    rollout_obs = []
    rollout_act = []
    while len(rollout_obs) < n_timesteps:
        done = False
        obs = env.reset()
        while not done and len(rollout_obs) < n_timesteps:
            act, _ = model.predict(obs)
            rollout_obs.append(obs)
            rollout_act.append(act)
            obs, _, done, _ = env.step(act)

    rollout_obs = np.array(rollout_obs)
    exp_obs = (n_timesteps,) + env.observation_space.shape
    assert rollout_obs.shape == exp_obs

    rollout_act = np.array(rollout_act)
    exp_act = (n_timesteps,) + env.action_space.shape
    assert rollout_act.shape == exp_act

    return rollout_obs, rollout_act

_basics1()
