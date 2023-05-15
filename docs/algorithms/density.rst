.. _density docs:

=============================
Density-based reward modeling
=============================

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
