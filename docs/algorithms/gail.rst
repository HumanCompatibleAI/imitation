.. _gail docs:

================================================
Generative Adversarial Imitation Learning (GAIL)
================================================

`GAIL <https://arxiv.org/abs/1606.03476>`_ learns a policy by simultaneously training it
with a discriminator that aims to distinguish expert trajectories against
trajectories from the learned policy.

Notes
-----
- GAIL paper: `Generative Adversarial Imitation Learning <https://arxiv.org/abs/1606.03476>`_

Example
=======

Detailed example notebook: `3_train_gail.ipynb <https://github.com/HumanCompatibleAI/imitation/blob/master/examples/3_train_gail.ipynb>`_

.. testcode::

    from stable_baselines3 import PPO
    from stable_baselines3.ppo import MlpPolicy
    import gym
    import seals
    from imitation.data import rollout
    from imitation.data.wrappers import RolloutInfoWrapper
    from imitation.algorithms.adversarial.gail import GAIL
    from imitation.rewards.reward_nets import BasicRewardNet
    from imitation.util.networks import RunningNorm
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv


    env = gym.make("seals/CartPole-v0")
    expert = PPO(policy=MlpPolicy, env=env, n_steps=64)
    expert.learn(1000)


    rollouts = rollout.rollout(
        expert,
        DummyVecEnv([lambda: RolloutInfoWrapper(gym.make("seals/CartPole-v0"))] * 5),
        rollout.make_sample_until(min_timesteps=None, min_episodes=60),
    )

    venv = DummyVecEnv([lambda: gym.make("seals/CartPole-v0")] * 8)
    learner = PPO(env=venv, policy=MlpPolicy)
    reward_net = BasicRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm
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

    learner_rewards_before_training, _ = evaluate_policy(
        learner, venv, 100, return_episode_rewards=True
    )
    gail_trainer.train(20000)
    learner_rewards_after_training, _ = evaluate_policy(
        learner, venv, 100, return_episode_rewards=True
    )

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
