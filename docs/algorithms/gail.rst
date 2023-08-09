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
    import seals  # noqa: F401  # needed to load "seals/" environments
    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.ppo import MlpPolicy

    from imitation.algorithms.adversarial.gail import GAIL
    from imitation.data import rollout
    from imitation.data.wrappers import RolloutInfoWrapper
    from imitation.policies.serialize import load_policy
    from imitation.rewards.reward_nets import BasicShapedRewardNet
    from imitation.util.networks import RunningNorm
    from imitation.util.util import make_vec_env

    SEED = 42

    env = make_vec_env(
        "seals/CartPole-v0",
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
        rollout.make_sample_until(min_timesteps=None, min_episodes=60),
        rng=np.random.default_rng(SEED),
    )

    learner = PPO(
        env=env,
        policy=MlpPolicy,
        batch_size=64,
        ent_coef=0.0,
        learning_rate=0.00001,
        n_epochs=1,
        seed=SEED,
    )
    reward_net = BasicShapedRewardNet(
        observation_space=env.observation_space,
        action_space=env.action_space,
        normalize_input_layer=RunningNorm,
    )
    gail_trainer = GAIL(
        demonstrations=rollouts,
        demo_batch_size=1024,
        gen_replay_buffer_capacity=2048,
        n_disc_updates_per_round=4,
        venv=env,
        gen_algo=learner,
        reward_net=reward_net,
    )

    # evaluate the learner before training
    env.seed(SEED)
    learner_rewards_before_training, _ = evaluate_policy(
        learner, env, 100, return_episode_rewards=True,
    )

    # train the learner and evaluate again
    gail_trainer.train(20000)
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
.. autoclass:: imitation.algorithms.adversarial.gail.GAIL
    :members:
    :inherited-members:
    :noindex:

.. autoclass:: imitation.algorithms.adversarial.common.AdversarialTrainer
    :members:
    :inherited-members:
    :noindex:
