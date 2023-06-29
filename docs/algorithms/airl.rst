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
    import gymnasium as gym
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

    SEED = 42

    env = make_vec_env(
        "seals:seals/CartPole-v0",
        rng=np.random.default_rng(SEED),
        n_envs=8,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],  # to compute rollouts
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
        rollout.make_sample_until(min_episodes=60),
        rng=np.random.default_rng(SEED),
    )

    learner = PPO(
        env=env,
        policy=MlpPolicy,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.0005,
        gamma=0.95,
        clip_range=0.1,
        vf_coef=0.1,
        n_epochs=5,
        seed=SEED,
    )
    reward_net = BasicShapedRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )
    airl_trainer = AIRL(
        demonstrations=rollouts,
        demo_batch_size=2048,
        gen_replay_buffer_capacity=512,
        n_disc_updates_per_round=16,
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
    )

    env.reset(seed=SEED)
    learner_rewards_before_training, _ = evaluate_policy(
        learner, env, 100, return_episode_rewards=True,
    )
    airl_trainer.train(20000)  # Train for 2_000_000 steps to match expert.
    env.seed(SEED)
    learner_rewards_after_training, _ = evaluate_policy(
        learner, env, 100, return_episode_rewards=True,
    )

    print("mean reward after training:", np.mean(learner_rewards_after_training))
    print("mean reward before training:", np.mean(learner_rewards_before_training))

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
