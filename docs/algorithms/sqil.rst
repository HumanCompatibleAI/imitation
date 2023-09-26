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

.. note::

    This implementation is based on the DQN implementation in Stable Baselines 3,
    which does not implement the soft Q-learning and therefore does not support
    continuous actions. Therefore, this implementation only supports discrete actions
    and the name "soft" Q-learning could be misleading.

Example
=======

Detailed example notebook: :doc:`../tutorials/8_train_sqil`

.. testcode::
    :skipif: skip_doctests

    import datasets
    import gymnasium as gym
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import DummyVecEnv

    from imitation.algorithms import sqil
    from imitation.data import huggingface_utils

    # Download some expert trajectories from the HuggingFace Datasets Hub.
    dataset = datasets.load_dataset("HumanCompatibleAI/ppo-CartPole-v1")
    rollouts = huggingface_utils.TrajectoryDatasetSequence(dataset["train"])

    sqil_trainer = sqil.SQIL(
        venv=DummyVecEnv([lambda: gym.make("CartPole-v1")]),
        demonstrations=rollouts,
        policy="MlpPolicy",
    )
    # Hint: set to 1_000_000 to match the expert performance.
    sqil_trainer.train(total_timesteps=1_000)
    reward, _ = evaluate_policy(sqil_trainer.policy, sqil_trainer.venv, 10)
    print("Reward:", reward)

.. testoutput::
    :hide:

    ...

API
===
.. autoclass:: imitation.algorithms.sqil.SQIL
    :members:
    :noindex:
