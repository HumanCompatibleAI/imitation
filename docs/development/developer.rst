.. _DevGuide:

Developer Guide
==================

This guide explains the library structure of imitation. The code is organized such that logically similar files
are grouped into a folder. We maintain the following modules in src/imitation:

- algorithms

- data

- envs

- policies

- rewards

- scripts

- util


Algorithms
----------

The base algorithm class implements the following two classes:

- ``BaseImitationAlgorithm``: Base class for all imitation algorithms. 

- ``DemonstrationAlgorithm``: Base class for all demonstration based algorithms like BC, IRL, etc. This class is inherited from ``BaseImitationAlgorithm``.
    Demonstration algorithms offers following methods and properties:
    - ``policy`` property that returns a policy imitating the demonstration data.

    - ``set_demonstrations()`` method that sets the demonstrations data for learning.

All of the algorithms provide the ``train()`` method for training an agent using the algorithm.

All the available algorithms are present in algorithms/ with each algorithm in a distinct file. 
Adversarial algorithms like AIRL and GAIL are present in algorithms/adversarial.


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


Scripts
-------
.. automodule:: imitation.scripts
    :noindex:

Util
----
.. automodule:: imitation.util
    :noindex: