import pickle
import gym
import seals  # noqa: F401

from imitation.algorithms import bc
from imitation.data import rollout
from examples.utils import render_a_trajectory_and_print_reward

env = gym.make("seals/CartPole-v0")
with open("tests/testdata/expert_models/cartpole_0/rollouts/final.pkl", "rb") as f:
    demonstrations = rollout.flatten_trajectories(pickle.load(f))
bc_trainer = bc.BC(observation_space=env.observation_space, action_space=env.action_space,
                   demonstrations=demonstrations)

print("Before behaviour cloning:")
render_a_trajectory_and_print_reward(env, bc_trainer.policy)
bc_trainer.train(n_epochs=1)
print("After behaviour cloning:")
render_a_trajectory_and_print_reward(env, bc_trainer.policy)
