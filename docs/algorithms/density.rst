.. _density docs:

=============================
Density-Based Reward Modeling
=============================

Density-based reward modeling is an inverse reinforcement learning (IRL) technique that assigns higher rewards
to states or state-action pairs that occur more frequently in an expert's demonstrations.
This variant utilizes `kernel density estimation <https://en.wikipedia.org/wiki/Kernel_density_estimation>`_
to model the underlying distribution of expert demonstrations.
It assigns rewards to states or state-action pairs based on their estimated log-likelihood
under the distribution of expert demonstrations.

The key intuition behind this method is to incentivize the agent to take actions
that resemble the expert's actions in similar states.

While this approach is relatively simple, it does have several drawbacks:

- It assumes that the expert demonstrations are representative of the expert's behavior, which may not always be true.
- It does not provide an interpretable reward function.
- The kernel density estimation is not well-suited for high-dimensional state-action spaces.

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

    imitation_trainer = PPO(
        ActorCriticPolicy,
        env,
        learning_rate=3e-4,
        gamma=0.95,
        ent_coef=1e-4,
        n_steps=2048
    )
    density_trainer = db.DensityAlgorithm(
        venv=env,
        rng=rng,
        demonstrations=rollouts,
        rl_algo=imitation_trainer,
        density_type=db.DensityType.STATE_ACTION_DENSITY,
        is_stationary=True,
        kernel="gaussian",
        kernel_bandwidth=0.4,
        standardise_inputs=True,
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

    density_trainer.train_policy(100)  # Train for 1_000_000 steps to approach the expert.

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
