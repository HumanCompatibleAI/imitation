=============================
Density-based reward modeling
=============================

Example
=======

Detailed example notebook: `7_train_density.ipynb <https://github.com/HumanCompatibleAI/imitation/blob/master/examples/7_train_density.ipynb>`_

.. testcode::

    import pprint

    from stable_baselines3 import PPO
    from stable_baselines3.common.policies import ActorCriticPolicy

    from imitation.algorithms import density as db
    from imitation.data import types
    from imitation.util import util

    env = util.make_vec_env("Pendulum-v1", 2)
    rollouts = types.load("../tests/testdata/expert_models/pendulum_0/rollouts/final.pkl")

    imitation_trainer = PPO(ActorCriticPolicy, env)
    density_trainer = db.DensityAlgorithm(
        venv=env,
        demonstrations=rollouts,
        rl_algo=imitation_trainer,
    )
    density_trainer.train()

    novice_stats = density_trainer.test_policy()
    pprint.pprint(novice_stats)
    novice_stats_im = density_trainer.test_policy(true_reward=False, n_trajectories=1)
    pprint.pprint(novice_stats_im)

    density_trainer.train_policy(100)

    good_stats = density_trainer.test_policy(n_trajectories=1)
    print("Trained stats:")
    pprint.pprint(good_stats)
    novice_stats_im = density_trainer.test_policy(true_reward=False)
    print("Trained stats (imitation reward function):")
    pprint.pprint(novice_stats_im)

.. testoutput::
    :hide:

    ...

API
===
.. autoclass:: imitation.algorithms.density.DensityAlgorithm
    :members:
    :inherited-members:
    :noindex:
