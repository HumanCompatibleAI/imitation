import gym
import tensorflow as tf
import tqdm

def idealize():
    # Initialize example traj
    env = gym.make("FrozenLake-v0")

    reward_true = TODO
    # TODO: Pickle an optimized solution to save time,
    # XXX: Make it easy to abstract out which planner/optimizer to use
    # to generate policies.
    traj_list_expert = generate_traj_reward(env, reward_true)

    # Initialize networks (train these)
    policy_net = PPO1(MlpPolicy, env, verbose=1)

    # Doubly parameterized reward net -- theta and phi.
    # Requirements:
    #  1. Contains two networks, the state(-action) reward parameterized by
    #     theta and the state reward shaper parameterized by phi.
    #  2. We can evaluate either of these inner networks and also the
    #     combined output f(s, a, s').
    reward_net = init_reward_net(env, state_only=True, discount_factor=0.9)

    # Initialize trainer and train.
    trainer = AIRLTrainer(env, traj_list_expert, policy_net, reward_net)
    trainer.train(n_epochs=1000)

    # Get unshaped reward (depends only on state and theta)
    # XXX: Make sure that rewards can depend on states and actions.
    reward_simple = reward_net.get_unshaped()

    # Test on other environment/Evaluate.
