======================
Command Line Interface
======================

Many features of the core library are accessible via the command line interface built
using the `Sacred <https://github.com/idsia/sacred>`_ package.

Sacred is used to configure and run the algorithms of the core library.
It is centered around the concept of `experiments <https://sacred.readthedocs.io/en/stable/experiment.html>`_
which are composed of reusable `ingredients <https://sacred.readthedocs.io/en/stable/ingredients.html>`_.
Each experiment and each ingredient has its own configuration namespace.
Named configurations are used to specify a coherent set of configuration values.
It is recommended to at least read the
`Sacred documentation about the command line interface <https://sacred.readthedocs.io/en/stable/command_line.html>`_.

The :py:mod:`scripts <imitation.scripts>` package contains a number of sacred experiments to either execute algorithms or perform utility tasks.
The most important :py:mod:`ingredients <imitation.scripts.ingredients>` for imitation learning are:

- :py:mod:`Environments <imitation.scripts.ingredients.environment>`
- :py:mod:`Expert Policies <imitation.scripts.ingredients.expert>`
- :py:mod:`Expert Demonstrations <imitation.scripts.ingredients.demonstrations>`
- :py:mod:`Reward Functions <imitation.scripts.ingredients.reward>`


Usage Examples
==============

Here we demonstrate some usage examples for the command line interface.
You can always find out all the configurable values by running:

.. code-block:: bash

    python -m imitation.scripts.<script> print_config

Run BC on the `CartPole-v1` environment with a pre-trained PPO policy as expert:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python -m imitation.scripts.train_imitation bc with \
        cartpole \
        demonstrations.n_expert_demos=50 \
        bc.train_kwargs.n_batches=2000 \
        expert.policy_type=ppo \
        expert.loader_kwargs.path=tests/testdata/expert_models/cartpole_0/policies/final/model.zip

50 expert demonstrations are sampled from the PPO policy that is included in the testdata folder.
2000 batches are enough to train a good policy.

Run DAgger on the `CartPole-v0` environment with a random policy as expert:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python -m imitation.scripts.train_imitation dagger with \
        cartpole \
        dagger.total_timesteps=2000 \
        demonstrations.n_expert_demos=10 \
        expert.policy_type=random

This will not produce any meaningful results, since a random policy is not a good expert.


Run AIRL on the `MountainCar-v0` environment with a expert from the HuggingFace model hub:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python -m imitation.scripts.train_adversarial airl with \
        seals_mountain_car \
        total_timesteps=50000 \
        expert.policy_type=ppo-huggingface \
        demonstrations.n_expert_demos=500

TODO: tweak above parameters to get good results


Run GAIL on the `seals/Swimmer-v0` (named config) environment with an ensemble of reward networks:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    xxx

Algorithm Scripts
=================

What script to call for which algorithm?

+---------------------------------+----------+--------------------------+
|  Algorithm                      | Script   |  command line arguments  |
+=================================+==========+==========================+
|  BC                             |  xxx     |  xxx                     |
+---------------------------------+----------+--------------------------+
| DAgger                          |  xxx     |  xxx                     |
+---------------------------------+----------+--------------------------+
| AIRL                            |  xxx     |  xxx                     |
+---------------------------------+----------+--------------------------+
| GAIL                            |  xxx     |  xxx                     |
+---------------------------------+----------+--------------------------+
| Preference Comparison           |  xxx     |  xxx                     |
+---------------------------------+----------+--------------------------+
| MCE IRL                         |  none    |  xxx                     |
+---------------------------------+----------+--------------------------+
| Density Based Reward Estimation |  none    |  xxx                     |
+---------------------------------+----------+--------------------------+


Utility Scripts
===============

+--------------------------------+----------+--------------------------+
| Functionality                  | Script   |  command line arguments  |
+================================+==========+==========================+
|Reinforcement Learning          |  xxx     |  xxx                     |
+--------------------------------+----------+--------------------------+
| Evaluating a Policy            |  xxx     |  xxx                     |
+--------------------------------+----------+--------------------------+
| Parallel Execution             |  xxx     |  xxx                     |
+--------------------------------+----------+--------------------------+
| Converting Trajectory Formats  |  xxx     |  xxx                     |
+--------------------------------+----------+--------------------------+
| Analyzing Experimental Results |  xxx     |  xxx                     |
+--------------------------------+----------+--------------------------+


Output Directories
==================

Where do all the files go and what is their purpose?