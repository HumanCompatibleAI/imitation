.. _DevGuide:

Developer Guide
==================

This guide explains the library structure of imitation. The code is organized such that logically similar files
are grouped into a subpackage. We maintain the following subpackages in ``src/imitation``:

- ``algorithms``: the core implementation of imitation and reward learning algorithms.

- ``data``: modules to collect, store and manipulate transitions and trajectories from RL environments.

- ``envs``: provides test environments.

- ``policies``: provides modules that define policies and methods to manipulate them (e.g., serialization).

- ``regularization``: implements a variety of regularization techniques for NN weights.

- ``rewards``: modules to build, serialize and preprocess neural network based reward functions.

- ``scripts``: command-line scripts for running experiments through Sacred.

- ``util``: provides utility functions like logging, configurations, etc.


Algorithms
----------

The ``imitation.algorithms.base`` module defines the following two classes:

- ``BaseImitationAlgorithm``: Base class for all imitation algorithms.

- | ``DemonstrationAlgorithm``: Base class for all demonstration-based algorithms like BC, IRL, etc. This class subclasses ``BaseImitationAlgorithm``.
  | Demonstration algorithms offer the following methods and properties:

  - ``policy`` property that returns a policy imitating the demonstration data.

  - ``set_demonstrations`` method that sets the demonstrations data for learning.

All of the algorithms provide the ``train`` method for training an agent and/or a reward network.

All the available algorithms are present in ``algorithms/`` with each algorithm in a distinct file.
Adversarial algorithms like AIRL and GAIL are present in ``algorithms/adversarial``.


Data
----
.. automodule:: imitation.data
    :noindex:

    ``data.wrapper.BufferingWrapper``: Wraps a vectorized environment ``VecEnv`` to save the trajectories from all the environments
    in a buffer.

    ``data.wrapper.RolloutInfoWrapper``: Wraps a ``gym.Env`` environment to log the original observations and rewards received from
    the environment. The observations and rewards of the entire episode are logged in the ``info`` dictionary with the
    key ``"rollout"``, in the final time step of the episode. This wrapper is useful for saving rollout trajectories, especially
    in cases where you want to bypass the reward and/or observation overrides from other wrappers.
    See ``data.rollout.unwrap_traj`` for details and ``scripts/train_rl.py`` for an example use case.

    ``data.rollout.rollout``: Generates rollout by taking in any policy as input along with the environment.

Policies
--------

The ``imitation.policies`` subpackage contains the following modules:

- ``policies.base``: defines commonly used policies across the library like ``FeedForward32Policy``, ``SAC1024Policy``, ``NormalizeFeaturesExtractor``, etc.
- ``policies.exploration_wrapper``: defines the ``ExplorationWrapper`` class that wraps a policy to create a partially randomized policy useful for exploration.
- ``policies.replay_buffer_wrapper``: defines the ``ReplayBufferRewardWrapper`` to wrap a replay buffer that returns transitions with rewards specified by a reward function.
- ``policies.serialize``: defines various functions to save and load serialized policies from the disk or the Hugging Face hub.

Regularization
--------------

The ``imitation.regularization`` subpackage provides an API for creating neural network regularizers. It provides classes such as
``regularizers.LpRegularizer`` and ``regularizers.WeightDecayRegularizer`` to regularize the loss function and the weights of
a network, respectively. The ``updaters.IntervalParamScaler`` class also provides support to scale the lambda hyperparameter
of a regularizer up when the ratio of validation to training loss is above an upper bound,
and scales it down when the ratio drops below a lower bound.

Rewards
-------

The ``imitation.rewards`` subpackage contains code related to building, serializing, and loading reward networks.
Some of the classes include:

- ``rewards.reward_nets.RewardNet``: is the base reward network class. Reward networks can take state, action, and the next state as input to predict the reward.
  The ``forward`` method is used while training the network, whereas the ``predict`` method is used during evaluation.

- ``rewards.reward_nets.BasicRewardNet``: builds a MLP reward network.

- ``rewards.reward_nets.CnnRewardNet``: builds a CNN based reward network.

- ``rewards.reward_nets.RewardEnsemble``: builds an ensemble of reward networks.

- ``rewards.reward_wrapper.RewardVecEnvWrapper``: This class wraps a ``VecEnv`` with a custom ``RewardFn``.
  The default reward function of the environment is overridden with the passed reward function,
  and the original rewards are stored in the ``info_dict`` with the ``original_env_rew`` key.
  This class is used to override the original reward function of an environment with a learned
  reward function from the reward learning algorithms like preference comparisons.

The ``imitation.rewards.serialize`` module contains functions to load serialized reward functions.

For more see the :ref:`Reward Networks Tutorial <reward-net docs>`.

Scripts
-------

We use Sacred to provide a command-line interface to run the experiments. The scripts to run the end-to-end experiments are
available in ``scripts/``. You can take a look at the following doc links to understand how to use Sacred:

- `Experiment Overview <https://sacred.readthedocs.io/en/stable/experiment.html>`_: Explains how to create and run experiments. Each script, defined in ``scripts/``, has a corresponding experiment object, defined in ``scripts/config``, with the experiment object and Python source files named after the algorithm(s) supported. For example, the ``train_rl_ex`` object is defined in ``scripts.config.train_rl`` and its main function is in ``scripts.train_rl``.

- `Ingredients <https://sacred.readthedocs.io/en/stable/ingredients.html>`_: Explains how to use ingredients to avoid code duplication across experiments. The ingredients used in our experiments are defined in ``scripts/ingredients/``:

  .. autosummary::
    imitation.scripts.ingredients.logging
    imitation.scripts.ingredients.demonstrations
    imitation.scripts.ingredients.environment
    imitation.scripts.ingredients.expert
    imitation.scripts.ingredients.reward
    imitation.scripts.ingredients.rl
    imitation.scripts.ingredients.policy
    imitation.scripts.ingredients.wb

- `Configurations <https://sacred.readthedocs.io/en/stable/configuration.html>`_: Explains how to use configurations to parametrize runs. The configurations for different algorithms are defined in their file in ``scripts/``. Some of the commonly used configs and ingredients used across algorithms are defined in ``scripts/ingredients/``.

- `Command-Line Interface <https://sacred.readthedocs.io/en/stable/command_line.html>`_: Explains how to run the experiments through the command-line interface. Also, note the section on how to `print configs <https://sacred.readthedocs.io/en/stable/command_line.html#print-config>`_ to verify the configurations used for the run.

- `Controlling Randomness <https://sacred.readthedocs.io/en/stable/randomness.html>`_: Explains how to control randomness by seeding experiments through Sacred.

Util
----

``imitation.util.logger.HierarchicalLogger``: A logger that supports contexts for accumulating the mean of values of all the logged keys.
The logger internally maintains one separate ``stable_baselines3.common.logger.Logger`` object for logging the mean values, and one ``Logger`` object for the raw values for each context.
The ``accumulate_means`` context cannot be called inside an already open ``accumulate_means`` context.
The ``imitation.util.logger.configure`` function can be used to easily construct a ``HierarchicalLogger`` object.

``imitation.util.networks``: This module provides some additional neural network layers that can be used for imitation like ``RunningNorm`` and ``EMANorm`` that normalize their inputs.
The module also provides functions like ``build_mlp`` and ``build_cnn`` to quickly build neural networks.

``imitation.util.util``: This module provides miscellaneous util functions like ``make_vec_env`` to easily construct vectorized environments and ``safe_to_tensor`` that converts a NumPy array to a PyTorch tensor.

``imitation.util.video_wrapper.VideoWrapper``: A wrapper to record rendered videos from an environment.
