.. _mce irl docs:

===============================================================
Maximum Causal Entropy Inverse Reinforcement Learning (MCE IRL)
===============================================================

Implements `Modeling Interaction via the Principle of Maximum Causal Entropy <https://www.cs.cmu.edu/~bziebart/publications/maximum-causal-entropy.pdf>`_.

Example
=======

Detailed example notebook: :doc:`../tutorials/6_train_mce`

.. testcode::
    :skipif: skip_doctests

    from functools import partial

    from seals import base_envs
    from seals.diagnostics.cliff_world import CliffWorldEnv
    import numpy as np

    from stable_baselines3.common.vec_env import DummyVecEnv

    from imitation.algorithms.mce_irl import (
        MCEIRL,
        mce_occupancy_measures,
        mce_partition_fh,
    )
    from imitation.data import rollout
    from imitation.rewards import reward_nets

    rng = np.random.default_rng(0)

    env_creator = partial(CliffWorldEnv, height=4, horizon=8, width=7, use_xy_obs=True)
    env_single = env_creator()

    state_env_creator = lambda: base_envs.ExposePOMDPStateWrapper(env_creator())

    # This is just a vectorized environment because `generate_trajectories` expects one
    state_venv = DummyVecEnv([state_env_creator] * 4)

    _, _, pi = mce_partition_fh(env_single)

    _, om = mce_occupancy_measures(env_single, pi=pi)

    reward_net = reward_nets.BasicRewardNet(
        env_single.observation_space,
        env_single.action_space,
        hid_sizes=[256],
        use_action=False,
        use_done=False,
        use_next_state=False,
    )

    # training on analytically computed occupancy measures
    mce_irl = MCEIRL(
        om,
        env_single,
        reward_net,
        log_interval=250,
        optimizer_kwargs={"lr": 0.01},
        rng=rng,
    )
    occ_measure = mce_irl.train()

    imitation_trajs = rollout.generate_trajectories(
        policy=mce_irl.policy,
        venv=state_venv,
        sample_until=rollout.make_min_timesteps(5000),
        rng=rng,
    )
    print("Imitation stats: ", rollout.rollout_stats(imitation_trajs))

.. testoutput::
    :hide:

    ...

API
===
.. autoclass:: imitation.algorithms.mce_irl.MCEIRL
    :members:
    :inherited-members:
    :noindex:

.. autoclass:: imitation.algorithms.base.DemonstrationAlgorithm
    :members:
    :noindex:
