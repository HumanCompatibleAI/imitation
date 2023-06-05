.. _gail docs:

================================================
Generative Adversarial Imitation Learning (GAIL)
================================================

`GAIL <https://arxiv.org/abs/1606.03476>`_ learns a policy by simultaneously training it
with a discriminator that aims to distinguish expert trajectories against
trajectories from the learned policy.

.. note::
    GAIL paper: `Generative Adversarial Imitation Learning <https://arxiv.org/abs/1606.03476>`_

Example
=======

Detailed example notebook: :doc:`../tutorials/3_train_gail`

.. testcode::
    :skipif: skip_doctests

    import numpy as np
    import gym
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.ppo import MlpPolicy

    from imitation.algorithms.adversarial.gail import GAIL
    from imitation.data import rollout
    from imitation.data.wrappers import RolloutInfoWrapper
    from imitation.rewards.reward_nets import BasicRewardNet
    from imitation.util.networks import RunningNorm
    from imitation.util.util import make_vec_env

    rng = np.random.default_rng(0)

    env = gym.make("seals/CartPole-v0")
    expert = PPO(policy=MlpPolicy, env=env, n_steps=64)
    expert.learn(1000)

    rollouts = rollout.rollout(
        expert,
        make_vec_env(
            "seals/CartPole-v0",
            n_envs=5,
            post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
            rng=rng,
        ),
        rollout.make_sample_until(min_timesteps=None, min_episodes=60),
        rng=rng,
    )

    venv = make_vec_env("seals/CartPole-v0", n_envs=8, rng=rng)
    learner = PPO(env=venv, policy=MlpPolicy)
    reward_net = BasicRewardNet(
        venv.observation_space,
        venv.action_space,
        normalize_input_layer=RunningNorm,
    )
    gail_trainer = GAIL(
        demonstrations=rollouts,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=4,
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
    )

    gail_trainer.train(20000)
    rewards, _ = evaluate_policy(learner, venv, 100, return_episode_rewards=True)
    print("Rewards:", rewards)

.. testoutput::
    :hide:

    ...

API
===
.. autoclass:: imitation.algorithms.adversarial.gail.GAIL
    :members:
    :inherited-members:
    :noindex:

.. autoclass:: imitation.algorithms.adversarial.common.AdversarialTrainer
    :members:
    :inherited-members:
    :noindex:
