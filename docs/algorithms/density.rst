=============================
Density-based reward modeling
=============================

Example
=======

Detailed example notebook: `7_train_density.ipynb <https://github.com/HumanCompatibleAI/imitation/blob/master/examples/7_train_density.ipynb>`_

.. testcode::

    import pprint
    from imitation.algorithms import density as db
    from imitation.data import types
    from imitation.util import util

    from stable_baselines3.common.policies import ActorCriticPolicy
    from stable_baselines3 import PPO

    env_name = "Pendulum-v1"
    env = util.make_vec_env(env_name, 8)
    rollouts = types.load("../tests/testdata/expert_models/pendulum_0/rollouts/final.pkl")


    imitation_trainer = PPO(ActorCriticPolicy, env)
    density_trainer = db.DensityAlgorithm(
        venv=env,
        demonstrations=rollouts,
        rl_algo=imitation_trainer,
        )
    density_trainer.train()

    novice_stats = density_trainer.test_policy()
    print("Novice stats (true reward function):")
    pprint.pprint(novice_stats)
    novice_stats_im = density_trainer.test_policy(
        true_reward=False, n_trajectories=10
    )
    print("Novice stats (imitation reward function):")
    pprint.pprint(novice_stats_im)

    for i in range(100):
        density_trainer.train_policy(1e5)

        good_stats = density_trainer.test_policy(n_trajectories=10)
        print(f"Trained stats (epoch {i}):")
        pprint.pprint(good_stats)
        novice_stats_im = density_trainer.test_policy(true_reward=False)
        print(f"Trained stats (imitation reward function, epoch {i}):")
        pprint.pprint(novice_stats_im)

API
===
.. autoclass:: imitation.algorithms.density.DensityAlgorithm
    :members:
    :inherited-members:
    :noindex:
