======================
Preference comparisons
======================

The preference comparisons algorithms learns a reward function from the preferences between pairs of 
trajectories. The reward function is learnt by gathering the preference between trajectories and 
training a reward network with the cross entropy loss to predict the trajectory that will get a higher reward.

Notes
-----
- Preference comparisons paper: `Deep Reinforcement Learning from Human Preferences <https://arxiv.org/pdf/1706.03741.pdf>`_

- An ensemble of reward networks can also be trained instead of a single network. The uncertainty in the preference between the member networks can be used to actively select preference queries.
    
Example
=======

Detailed example notebook: `5_train_preference_comparisons.ipynb <https://github.com/HumanCompatibleAI/imitation/blob/master/examples/5_train_preference_comparisons.ipynb>`_

.. testcode::

    from stable_baselines3 import PPO
    from stable_baselines3.common.evaluation import evaluate_policy
    from stable_baselines3.ppo import MlpPolicy

    from imitation.algorithms import preference_comparisons
    from imitation.policies.base import FeedForward32Policy, NormalizeFeaturesExtractor
    from imitation.rewards.reward_nets import BasicRewardNet
    from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
    from imitation.util.networks import RunningNorm
    from imitation.util.util import make_vec_env

    venv = make_vec_env("Pendulum-v1")

    reward_net = BasicRewardNet(
        venv.observation_space, venv.action_space, normalize_input_layer=RunningNorm,
    )

    fragmenter = preference_comparisons.RandomFragmenter(warning_threshold=0, seed=0)
    gatherer = preference_comparisons.SyntheticGatherer(seed=0)
    preference_model = preference_comparisons.PreferenceModel(reward_net)
    reward_trainer = preference_comparisons.BasicRewardTrainer(
        model=reward_net,
        loss=preference_comparisons.CrossEntropyRewardLoss(preference_model),
        epochs=3,
    )

    agent = PPO(
        policy=FeedForward32Policy,
        policy_kwargs=dict(
            features_extractor_class=NormalizeFeaturesExtractor,
            features_extractor_kwargs=dict(normalize_class=RunningNorm),
        ),
        env=venv,
        n_steps=2048 // venv.num_envs,
    )

    trajectory_generator = preference_comparisons.AgentTrainer(
        algorithm=agent,
        reward_fn=reward_net,
        venv=venv,
        exploration_frac=0.0,
        seed=0,
    )

    pref_comparisons = preference_comparisons.PreferenceComparisons(
        trajectory_generator,
        reward_net,
        num_iterations=5,
        fragmenter=fragmenter,
        preference_gatherer=gatherer,
        reward_trainer=reward_trainer,
        seed=0,
        initial_epoch_multiplier=1,
    )
    pref_comparisons.train(total_timesteps=5_000, total_comparisons=200)

    learned_reward_venv = RewardVecEnvWrapper(venv, reward_net.predict)
    learner = PPO(
        policy=MlpPolicy,
        env=learned_reward_venv,
        n_steps=64,
    )
    learner.learn(1000)

    reward, _ = evaluate_policy(learner.policy, venv, 10)
    print("Reward:", reward)

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
