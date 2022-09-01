===============================================================
Maximum Causal Entropy Inverse Reinforcement Learning (MCE IRL)
===============================================================

Implements `Modeling Interaction via the Principle of Maximum Causal Entropy <https://www.cs.cmu.edu/~bziebart/publications/maximum-causal-entropy.pdf>`_.

Example
=======

Detailed example notebook: `6_train_mce.ipynb <https://github.com/HumanCompatibleAI/imitation/blob/master/examples/6_train_mce.ipynb>`_

.. testcode::

    from functools import partial

    from stable_baselines3.common.vec_env import DummyVecEnv

    from imitation.algorithms.mce_irl import (
        MCEIRL,
        mce_occupancy_measures,
        mce_partition_fh,
    )
    from imitation.data import rollout
    from imitation.envs import resettable_env
    from imitation.envs.examples.model_envs import CliffWorld
    from imitation.rewards import reward_nets

    env_creator = partial(CliffWorld, height=4, horizon=8, width=7, use_xy_obs=True)
    env_single = env_creator()

    # This is just a vectorized environment because `generate_trajectories` expects one
    state_venv = resettable_env.DictExtractWrapper(DummyVecEnv([env_creator] * 4), "state")

    _, _, pi = mce_partition_fh(env_single)

    _, om = mce_occupancy_measures(env_single, pi=pi)

    reward_net = reward_nets.BasicRewardNet(
        env_single.pomdp_observation_space,
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
    )
    occ_measure = mce_irl.train()

    imitation_trajs = rollout.generate_trajectories(
        policy=mce_irl.policy,
        venv=state_venv,
        sample_until=rollout.make_min_timesteps(5000),
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
