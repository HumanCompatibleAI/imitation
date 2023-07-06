======================
Command Line Interface
======================

Many features of the core library are accessible via the command line interface built
using the `Sacred <https://github.com/idsia/sacred>`_ package.

Sacred is used to configure and run the algorithms.
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

Note: the cartpole environment is specified via a named configuration

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
        total_timesteps=5000 \
        expert.policy_type=ppo-huggingface \
        demonstrations.n_expert_demos=500

Note: the small number of total timesteps is only for demonstration purposes and will not produce a good policy.


Run GAIL on the `seals/Swimmer-v0` environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we do not use the named configuration for the seals environment, but instead specify the gym_id directly.
The :code:`seals:` prefix ensures that the seals package is imported and the environment is registered.

Note that the Swimmer environment needs `mujoco_py` to be installed.

.. code-block:: bash

    python -m imitation.scripts.train_adversarial gail with \
            environment.gym_id="seals:seals/Swimmer-v0" \
            total_timesteps=5000 \
            demonstrations.n_expert_demos=50


Algorithm Scripts
=================

Call the algorithm scripts like this:

.. code-block:: bash

    python -m imitation.scripts.<script> [command] with <named_config> <config_values>

+---------------------------------+------------------------------+----------+
|  algorithm                      | script                       |  command |
+=================================+==============================+==========+
| BC                              | train_imitation              |  bc      |
+---------------------------------+------------------------------+----------+
| DAgger                          | train_imitation              |  dagger  |
+---------------------------------+------------------------------+----------+
| AIRL                            | train_adversarial            |  airl    |
+---------------------------------+------------------------------+----------+
| GAIL                            | train_adversarial            |  gail    |
+---------------------------------+------------------------------+----------+
| Preference Comparison           | train_preference_comparisons |  -       |
+---------------------------------+------------------------------+----------+
| MCE IRL                         | none                         |  -       |
+---------------------------------+------------------------------+----------+
| Density Based Reward Estimation | none                         |  -       |
+---------------------------------+------------------------------+----------+

Utility Scripts
===============

Call the utility scripts like this:

.. code-block:: bash

    python -m imitation.scripts.<script>

+-----------------------------------------+-----------------------------------------------------------+
| Functionality                           | Script                                                    |
+=========================================+===========================================================+
| Reinforcement Learning                  | :py:mod:`train_rl <imitation.scripts.train_rl>`           |
+-----------------------------------------+-----------------------------------------------------------+
| Evaluating a Policy                     | :py:mod:`eval_policy <imitation.scripts.eval_policy>`     |
+-----------------------------------------+-----------------------------------------------------------+
| Parallel Execution of Algorithm Scripts | :py:mod:`parallel <imitation.scripts.parallel>`           |
+-----------------------------------------+-----------------------------------------------------------+
| Converting Trajectory Formats           | :py:mod:`convert_trajs <imitation.scripts.convert_trajs>` |
+-----------------------------------------+-----------------------------------------------------------+
| Analyzing Experimental Results          | :py:mod:`analyze <imitation.scripts.analyze>`             |
+-----------------------------------------+-----------------------------------------------------------+


Output Directories
==================

The results of the script runs are stored in the following directory structure:

.. code-block::

    output
    ├── <algo>
    │   └── <environment>
    │       └── <timestamp>
    │           ├── log
    │           ├── monitor
    │           └── sacred -> ../../../sacred/<script_name>/1
    └── sacred
        └── <script_name>
            ├── 1
            └── _sources

It contains the final model, tensorboard logs, sacred logs and the sacred source files.
