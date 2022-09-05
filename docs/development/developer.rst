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

- ``scripts``: command-line scripts for running experiments through Sacred.

- ``util``: provides utility functions like logging, configurations, etc.


Algorithms
----------

The ``imitation.algorithms.base`` module defines the following two classes:

- ``BaseImitationAlgorithm``: Base class for all imitation algorithms. 

- | ``DemonstrationAlgorithm``: Base class for all demonstration based algorithms like BC, IRL, etc. This class subclasses ``BaseImitationAlgorithm``. 
  | Demonstration algorithms offer the following methods and properties:

  - ``policy`` property that returns a policy imitating the demonstration data.

  - ``set_demonstrations()`` method that sets the demonstrations data for learning.

All of the algorithms provide the ``train()`` method for training an agent and/or a reward network.

All the available algorithms are present in ``algorithms/`` with each algorithm in a distinct file. 
Adversarial algorithms like AIRL and GAIL are present in ``algorithms/adversarial``.


Data
----
.. automodule:: imitation.data
    :noindex:

    ``data.wrapper.BufferingWrapper``: Wraps a vectorized environment ``VecEnv`` to save the trajectories from all the environments
    in a buffer.

    ``data.wrapper.RolloutInfoWrapper``: Wraps a ``gym.Env`` environment to log the original observations and rewards recieved from 
    the environment. The observations and rewards of the entire episode are logged in the ``info`` dictionary with the
    key ``"rollout"``, in the final time step of the episode. This wrapper is useful for saving rollout trajectories, especially
    in cases where you want to bypass the reward and/or observation overrides from other wrappers. 
    See ``data.rollout.unwrap_traj`` for details and ``scripts/train_rl.py`` for an example use case.

    ``data.rollout.rollout``: Generates rollout by taking in any policy as input along with the environment. 

.. Policies
.. --------
.. .. automodule:: imitation.policies
..     :noindex:

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

We use Sacred to provide a command-line interface to run the experiments. The scripts to run the end-to-end experiments are
available in ``scripts/``. You can take a look at the following doc links to understand how to use Sacred:

- `Experiment Overview <https://sacred.readthedocs.io/en/stable/experiment.html>`_: Explains how to create and run experiments. Each script, defined in ``scripts/``, has a corresponding experiment object, defined in ``scripts/config``, with the experiment object and Python source files named after the algorithm(s) supported. For example, the ``train_rl_ex`` object is defined in ``scripts.config.train_rl`` and its main function is in ``scripts.train_rl``.

- `Ingredients <https://sacred.readthedocs.io/en/stable/ingredients.html>`_: Explains how to use ingredients to avoid code duplication across experiments. The ingredients used in our experiments are defined in ``scripts/common/``:
  
  .. autosummary:: 
    imitation.scripts.common.common
    imitation.scripts.common.demonstrations
    imitation.scripts.common.reward
    imitation.scripts.common.rl
    imitation.scripts.common.train
    imitation.scripts.common.wb

- `Configurations <https://sacred.readthedocs.io/en/stable/configuration.html>`_: Explains how to use configurations to parametrize runs. The configurations for different algorithms are defined in their file in ``scripts/``. Some of the commonly used configs and ingredients used across algorithms are defined in ``scripts/common/``.

- `Command-Line Interface <https://sacred.readthedocs.io/en/stable/command_line.html>`_: Explains how to run the experiments through the command-line interface. Also note the section on how to `print configs <https://sacred.readthedocs.io/en/stable/command_line.html#print-config>`_ to verify the configurations used for the run.

- `Controlling Randomness <https://sacred.readthedocs.io/en/stable/randomness.html>`_: Explains how to control randomness by seeding experiments through Sacred.

.. Util
.. ----
.. .. automodule:: imitation.util
..     :noindex:

