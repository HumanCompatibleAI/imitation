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
    import seals  # noqa: F401  # needed to load "seals/" environments
    from imitation.policies.serialize import load_policy
    from imitation.util.util import make_vec_env
    from imitation.data.wrappers import RolloutInfoWrapper

    rng = np.random.default_rng(0)
    env = make_vec_env(
        "seals/CartPole-v0",
        rng=rng,
        n_envs=1,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # for computing rollouts
    )
    expert = load_policy(
        "ppo-huggingface",
        organization="HumanCompatibleAI",
        env_name="seals-CartPole-v0",
        venv=env,
    )
    rollouts = rollout.rollout(
        expert,
        env,
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
