.. _preference comparisons docs:

======================
Preference Comparisons
======================

The preference comparison algorithm learns a reward function from preferences between pairs of trajectories.
The comparisons are modeled as being generated from a Bradley-Terry (or Boltzmann rational) model,
where the probability of preferring trajectory A over B is proportional to the exponential of the
difference between the return of trajectory A minus B. In other words, the difference in returns forms a logit
for a binary classification problem, and accordingly the reward function is trained using a cross-entropy loss
to predict the preference comparison.

.. note::
    - Our implementation is based on the  `Deep Reinforcement Learning from Human Preferences <https://arxiv.org/pdf/1706.03741.pdf>`_ algorithm.
    - An ensemble of reward networks can also be trained instead of a single network. The uncertainty in the preference between the member networks can be used to actively select preference queries.

Example
=======

You can copy this example to train `PPO <https://stable-baselines3.readthedocs.io/en/master/modules/ppo.html>`_ on `Pendulum <https://www.gymlibrary.dev/environments/classic_control/pendulum/>`_ using a reward model trained on 200 synthetic preference comparisons.
For a more detailed example, refer to :doc:`../tutorials/5_train_preference_comparisons`.

.. testcode::
    :skipif: skip_doctests

    import numpy as np

    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.ppo import MlpPolicy

    from imitation.algorithms import preference_comparisons
    from imitation.policies.base import FeedForward32Policy, NormalizeFeaturesExtractor
    from imitation.rewards.reward_nets import BasicRewardNet
    from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
    from imitation.util.networks import RunningNorm
    from imitation.util.util import make_vec_env

    rng = np.random.default_rng(0)

    venv = make_vec_env("Pendulum-v1", rng=rng)

    reward_net = BasicRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm,
    )

    fragmenter = preference_comparisons.RandomFragmenter(warning_threshold=0, rng=rng)
    gatherer = preference_comparisons.SyntheticGatherer(rng=rng)
    querent = preference_comparisons.PreferenceQuerent()
    preference_model = preference_comparisons.PreferenceModel(reward_net)
    reward_trainer = preference_comparisons.BasicRewardTrainer(
        preference_model=preference_model,
        loss=preference_comparisons.CrossEntropyRewardLoss(),
        epochs=10,
        rng=rng,
    )

    agent = PPO(
        policy=FeedForward32Policy,
        policy_kwargs=dict(
            features_extractor_class=NormalizeFeaturesExtractor,
            features_extractor_kwargs=dict(normalize_class=RunningNorm),
        ),
        env=venv,
        n_steps=2048 // venv.num_envs,
        clip_range=0.1,
        ent_coef=0.01,
        gae_lambda=0.95,
        n_epochs=10,
        gamma=0.97,
        learning_rate=2e-3,
    )

    trajectory_generator = preference_comparisons.AgentTrainer(
        algorithm=agent,
        reward_fn=reward_net,
        venv=venv,
        exploration_frac=0.05,
        rng=rng,
    )

    pref_comparisons = preference_comparisons.PreferenceComparisons(
        trajectory_generator,
        reward_net,
        num_iterations=5, # Set to 60 for better performance
        fragmenter=fragmenter,
        preference_querent=querent,
        preference_gatherer=gatherer,
        reward_trainer=reward_trainer,
        initial_epoch_multiplier=4,
        initial_comparison_frac=0.1,
        query_schedule="hyperbolic",
    )
    pref_comparisons.train(total_timesteps=50_000, total_comparisons=200)

    n_eval_episodes = 10
    reward_mean, reward_std = evaluate_policy(agent.policy, venv, n_eval_episodes)
    reward_stderr = reward_std/np.sqrt(n_eval_episodes)
    print(f"Reward: {reward_mean:.0f} +/- {reward_stderr:.0f}")

.. testoutput::
    :hide:

    ...

API
===
.. autoclass:: imitation.algorithms.preference_comparisons.PreferenceComparisons
    :members:
    :inherited-members:
    :noindex:

.. autoclass:: imitation.algorithms.base.BaseImitationAlgorithm
    :members:
    :noindex:
