"""This is a simple example demonstrating behavior cloning."""
import pickle

import gym
import seals  # noqa: F401
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.data import rollout

env = gym.make("seals/CartPole-v0")
with open(".pytest_cache/d/experts/CartPole-v1/rollout.pkl", "rb") as f:
    demonstrations = rollout.flatten_trajectories(pickle.load(f))
bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    demonstrations=demonstrations,
)

print("Before behaviour cloning:")
reward, _ = evaluate_policy(bc_trainer.policy, env, 3, render=True)
print("Untrained Reward:", reward)
bc_trainer.train(n_epochs=1)
print("After behaviour cloning:")
reward, _ = evaluate_policy(bc_trainer.policy, env, 3, render=True)
print("Trained Reward:", reward)
