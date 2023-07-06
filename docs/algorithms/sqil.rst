.. _soft q imitation learning docs:

================================
Soft Q Imitation Learning (SQIL)
================================

Soft Q Imitation learning learns to imitate a policy from demonstrations by
using the DQN algorithm with modified rewards. During each policy update, half
of the batch is sampled from the demonstrations and half is sampled from the
environment. Expert demonstrations are assigned a reward of 1, and the
environment is assigned a reward of 0. This encourages the policy to imitate
the demonstrations, and to simultaneously avoid states not seen in the
demonstrations.

Example
=======

Detailed example notebook: :doc:`../tutorials/10_train_sqil`

.. testcode::
    :skipif: skip_doctests

    import numpy as np
    import gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.ppo import MlpPolicy

    from imitation.algorithms import sqil
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

    sqil_trainer = sqil.SQIL(
        venv=DummyVecEnv([lambda: env]),
        demonstrations=transitions,
        policy="MlpPolicy",
    )
    sqil_trainer.train(total_timesteps=1000)
    reward, _ = evaluate_policy(sqil_trainer.policy, env, 10)
    print("Reward:", reward)

.. testoutput::
    :hide:

    ...

API
===
.. autoclass:: imitation.algorithms.sqil.SQIL
    :members:
    :noindex:
