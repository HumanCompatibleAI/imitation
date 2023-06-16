.. _density docs:

=============================
Density-based reward modeling
=============================

Density-based reward modeling is an inverse reinforcement learning (IRL) technique
that eliminates the need for explicit rewards by assigning higher rewards to states or state-action pairs
that occur more frequently in the expert's behavior.
This variant leverages kernel density estimation to model the underlying distribution of expert demonstrations,
enabling the generation of a reward model that captures the expert's preferences solely based on their observed
behavior.

The key intuition is that the expert prefers state-action pairs that occur more frequently,
so a reward function based on density will incentivize the agent to take similar actions as the expert.
The pros of this method are that it is simple and model-free.
The cons are that it assumes the density is an indicator of reward, which may not always be true.
It also does not provide a interpretable reward function.
Also, the kernel density estimation is not suited for high-dimensional state-action spaces.

Example
=======

Detailed example notebook: :doc:`../tutorials/7_train_density`

.. testcode::
    :skipif: skip_doctests

    import pprint
    import numpy as np

    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import ActorCriticPolicy

    from imitation.algorithms import density as db
    from imitation.data import serialize
    from imitation.util import util

    rng = np.random.default_rng(0)

    env = util.make_vec_env("Pendulum-v1", rng=rng, n_envs=2)
    rollouts = serialize.load("../tests/testdata/expert_models/pendulum_0/rollouts/final.npz")

    imitation_trainer = PPO(ActorCriticPolicy, env)
    density_trainer = db.DensityAlgorithm(
        venv=env,
        demonstrations=rollouts,
        rl_algo=imitation_trainer,
        rng=rng,
    )
    density_trainer.train()

    def print_stats(density_trainer, n_trajectories):
        stats = density_trainer.test_policy(n_trajectories=n_trajectories)
        print("True reward function stats:")
        pprint.pprint(stats)
        stats_im = density_trainer.test_policy(true_reward=False, n_trajectories=n_trajectories)
        print("Imitation reward function stats:")
        pprint.pprint(stats_im)

    print("Stats before training:")
    print_stats(density_trainer, 1)

    density_trainer.train_policy(100)

    print("Stats after training:")
    print_stats(density_trainer, 1)

.. testoutput::
    :hide:

    ...

API
===
.. autoclass:: imitation.algorithms.density.DensityAlgorithm
    :members:
    :inherited-members:
    :noindex:
