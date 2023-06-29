.. _behavioral cloning docs:

=======================
Behavioral Cloning (BC)
=======================

Behavioral cloning directly learns a policy by using supervised learning on
observation-action pairs from expert demonstrations. It is a simple approach to learning
a policy, but the policy often generalizes poorly and does not recover well from errors.

Alternatives to behavioral cloning include :ref:`DAgger <dagger docs>` (similar but gathers
on-policy demonstrations) and :ref:`GAIL <gail docs>`/:ref:`AIRL <airl docs>` (more robust
approaches to learning from demonstrations).

Example
=======

Detailed example notebook: :doc:`../tutorials/1_train_bc`

.. testcode::
    :skipif: skip_doctests

    import numpy as np
    import gymnasium as gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.ppo import MlpPolicy

    from imitation.algorithms import bc
    from imitation.data import rollout
    from imitation.data.wrappers import RolloutInfoWrapper

    rng = np.random.default_rng(0)
    env = gym.make("CartPole-v1")
    expert = PPO(policy=MlpPolicy, env=env)
    expert.learn(1000)

    rollouts = rollout.rollout(
        expert,
        DummyVecEnv([lambda: RolloutInfoWrapper(env)]),
        rollout.make_sample_until(min_timesteps=None, min_episodes=50),
        rng=rng,
    )
    transitions = rollout.flatten_trajectories(rollouts)

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        demonstrations=transitions,
        rng=rng,
    )
    bc_trainer.train(n_epochs=1)
    reward, _ = evaluate_policy(bc_trainer.policy, env, 10)
    print("Reward:", reward)

.. testoutput::
    :hide:

    ...

API
===
.. autoclass:: imitation.algorithms.bc.BC
    :members:
    :noindex:
