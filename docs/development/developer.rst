.. _DevGuide:

Developer Guide
==================

This guide explains the library structure of imitation. The code is organized such that logically similar files
are grouped into a folder. We maintain the following modules in ``src/imitation``:

- ``algorithms``: the core implementation of imitation and reward learning algorithms.

- ``data``: modules to collect, store and manipulate transition and trajectories from RL environments.

- ``envs``: provides test environments.

- ``policies``: modules defining policies and methods to manipulate them (e.g. serialization).

- ``rewards``: modules to build, serialize and preprocess neural network based reward functions.

- ``scripts``: command-line scripts for running experiments through sacred.

- ``util``: provides utility functions like logging, configurations, etc.


Algorithms
----------

The ``imitation.algorithms.base`` module defines the following two classes:

- ``BaseImitationAlgorithm``: Base class for all imitation algorithms. 

- | ``DemonstrationAlgorithm``: Base class for all demonstration based algorithms like BC, IRL, etc. This class subclasses ``BaseImitationAlgorithm``. 
  | Demonstration algorithms offers following methods and properties:

    - ``policy`` property that returns a policy imitating the demonstration data.

    - ``set_demonstrations()`` method that sets the demonstrations data for learning.

All of the algorithms provide the ``train()`` method for training an agent and/or reward network.

All the available algorithms are present in ``algorithms/`` with each algorithm in a distinct file. 
Adversarial algorithms like AIRL and GAIL are present in ``algorithms/adversarial``.


Data
----
.. automodule:: imitation.data
    :noindex:

Envs
----
.. automodule:: imitation.envs
    :noindex:

Policies
--------
.. automodule:: imitation.policies
    :noindex:

Rewards
-------
.. automodule:: imitation.rewards
    :noindex:

    ``rewards.reward_wrapper.RewardVecEnvWrapper``: This class wraps a ``VecEnv`` with a custom ``RewardFn``. 
    The default reward function of the environment is overridden with the passed reward function 
    and the original rewards are stored in the ``info_dict`` with the ``original_env_rew`` key. 
    This class is used to override the original reward function of an environment with a learned 
    reward function from the reward learning algorithms like preference comparisons.

Scripts
-------
.. automodule:: imitation.scripts
    :noindex:

Util
----
.. automodule:: imitation.util
    :noindex: