.. _airl docs:

=================================================
Adversarial Inverse Reinforcement Learning (AIRL)
=================================================
`AIRL <https://arxiv.org/abs/1710.11248>`_, similar to :ref:`GAIL <gail docs>`,
adversarially trains a policy against a discriminator that aims to distinguish the expert
demonstrations from the learned policy. Unlike GAIL, AIRL recovers a reward function
that is more generalizable to changes in environment dynamics.

The expert policy must be stochastic.


.. note::
    AIRL paper: `Learning Robust Rewards with Adversarial Inverse Reinforcement Learning <https://arxiv.org/abs/1710.11248>`_

Example
=======

Detailed example notebook: :doc:`../tutorials/4_train_airl`

.. testcode::
    :skipif: skip_doctests

    import numpy as np
    import gym
    import seals  # needed to load "seals/" environments
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.ppo import MlpPolicy

    from imitation.algorithms.adversarial.airl import AIRL
    from imitation.data import rollout
    from imitation.data.wrappers import RolloutInfoWrapper
    from imitation.policies.serialize import load_policy
    from imitation.rewards.reward_nets import BasicShapedRewardNet
    from imitation.util.networks import RunningNorm
    from imitation.util.util import make_vec_env

    rng = np.random.default_rng(0)

    env = gym.make("seals/CartPole-v0")
    expert = load_policy(
        "ppo-huggingface",
        organization="HumanCompatibleAI",
        env_name="seals-CartPole-v0",
        venv=env,
    )

    rollouts = rollout.rollout(
        expert,
        make_vec_env(
            "seals/CartPole-v0",
            rng=rng,
            n_envs=5,
            post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
        ),
        rollout.make_sample_until(min_timesteps=None, min_episodes=60),
        rng=rng,
    )

    venv = make_vec_env("seals/CartPole-v0", rng=rng, n_envs=8)
    learner = PPO(env=venv, policy=MlpPolicy)
    reward_net = BasicShapedRewardNet(
        venv.observation_space,
        venv.action_space,
        normalize_input_layer=RunningNorm,
    )
    airl_trainer = AIRL(
        demonstrations=rollouts,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
    )
    airl_trainer.train(20000)
    rewards, _ = evaluate_policy(learner, venv, 100, return_episode_rewards=True)
    print("Rewards:", rewards)

.. testoutput::
    :hide:

    ...

API
===
.. autoclass:: imitation.algorithms.adversarial.airl.AIRL
    :members:
    :inherited-members:
    :noindex:

.. autoclass:: imitation.algorithms.adversarial.common.AdversarialTrainer
    :members:
    :noindex:
