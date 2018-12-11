"""
MVP-not-really
"""

import gym
from stable_baselines.common.policies import MlpPolicy
from stable_baselines.common.vec_env import DummyVecEnv
from stable_baselines import PPO1

def _frozen_lake():
    """
    * Load frozen lake environment.
    * Train a PPO policy on frozen lake.
    * Generate 9 trajectory rollouts.
    * Print the state action pairs.
    BONUS:
    # Pickle the policy.
    # Load the policy.
    # Repeat the procedure above from the rollout part.
    """
    env = gym.make("FrozenLake-v0")
    env = DummyVecEnv([lambda: env])

    model = PPO1(MlpPolicy, env, verbose=1)
    model.learn(50000)

    for i in range(2):
        done = False
        obs = env.reset()
        while not done:
            a, _ = model.predict(obs)
            obs, reward, done, info = env.step(a)
            env.render()

_frozen_lake()
