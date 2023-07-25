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

Run BC on the ``CartPole-v1`` environment with a pre-trained PPO policy as expert:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. note:: Here the cartpole environment is specified via a named configuration.

.. code-block:: bash

    python -m imitation.scripts.train_imitation bc with \
        cartpole \
        demonstrations.n_expert_demos=50 \
        bc.train_kwargs.n_batches=2000 \
        expert.policy_type=ppo \
        expert.loader_kwargs.path=tests/testdata/expert_models/cartpole_0/policies/final/model.zip

50 expert demonstrations are sampled from the PPO policy that is included in the testdata folder.
2000 batches are enough to train a good policy.

Run DAgger on the ``CartPole-v0`` environment with a random policy as expert:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python -m imitation.scripts.train_imitation dagger with \
        cartpole \
        dagger.total_timesteps=2000 \
        demonstrations.n_expert_demos=10 \
        expert.policy_type=random

This will not produce any meaningful results, since a random policy is not a good expert.


Run AIRL on the ``MountainCar-v0`` environment with a expert from the HuggingFace model hub:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. code-block:: bash

    python -m imitation.scripts.train_adversarial airl with \
        seals_mountain_car \
        total_timesteps=5000 \
        expert.policy_type=ppo-huggingface \
        demonstrations.n_expert_demos=500

.. note:: The small number of total timesteps is only for demonstration purposes and will not produce a good policy.


Run GAIL on the ``seals/Swimmer-v0`` environment
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Here we do not use the named configuration for the seals environment, but instead specify the gym_id directly.
The ``seals:`` prefix ensures that the seals package is imported and the environment is registered.

.. note:: The Swimmer environment needs `mujoco_py` to be installed.

.. code-block:: bash

    python -m imitation.scripts.train_adversarial gail with \
            environment.gym_id="seals:seals/Swimmer-v0" \
            total_timesteps=5000 \
            demonstrations.n_expert_demos=50


Train an expert and save the rollouts explicitly, then train a policy on the saved rollouts:
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

First, train an expert and save the demonstrations. 
Note that by default these are saved in ``<log_dir>/rollouts/final.npz``.
Where for this script by default ``<log_dir>`` is ``output/train_rl/<environment>/<timestamp>`` .  
However, we can pass an explicit path as logging directory. 
By default, this will use ``ppo``. 

.. code-block:: bash

        python -m imitation.scripts.train_rl with pendulum \
                logging.log_dir=output/train_rl/Pendulum-v1/my_run  \

Now we can run the imitation sript (in this case DAgger) and pass the path to the demonstrations we just generated

.. code-block:: bash

        python -m imitation.scripts.train_imitation dagger with \
                pendulum \
                dagger.total_timesteps=2000 \
                demonstrations.source=local \
                demonstrations.path=output/train_rl/Pendulum-v1/my_run/rollouts/final.npz   


Visualise saved policies
^^^^^^^^^^^^^^^^^^^^^^^^ 
We can use the ``eval_policy`` script to visualise and render a saved policy. 
Here we are looking at the policy saved by the previous example. 

.. code-block:: bash

    python -m imitation.scripts.eval_policy with \
            expert.policy_type=ppo \
            expert.loader_kwargs.path=output/train_rl/Pendulum-v1/my_run/policies/final/model.zip \
            environment.num_vec=1 \
            render=True \
            environment.gym_id='Pendulum-v1' 



Comparing algorithm performances 
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^    
Let's use the cli to compare the performances of different algorithms. 

First, let's train an expert on the ``CartPole-v1`` environment.

.. code-block:: bash

    python -m imitation.scripts.train_rl with \
            cartpole \
            logging.log_dir=output/train_rl/CartPole-v1/expert \
            total_timesteps=10000

Now let's train a weaker agent. 

.. code-block:: bash

    python -m imitation.scripts.train_rl with \
        cartpole \
        logging.log_dir=output/train_rl/CartPole-v1/non_expert \
        total_timesteps=1000     # simply training less


We can evaluate each policy using the ``eval_policy`` script.
For the expert: 

.. code-block:: bash

    python -m imitation.scripts.eval_policy with \
            expert.policy_type=ppo \
            expert.loader_kwargs.path=output/train_rl/CartPole-v1/expert/policies/final/model.zip \
            environment.gym_id='CartPole-v1' \
            environment.num_vec=1 \
            logging.log_dir=output/eval_policy/CartPole-v1/expert

which will return something like 

.. code-block:: bash

    INFO - eval_policy - Result: {
            'n_traj': 74, 
            'monitor_return_len': 74, 
            'return_min': 26.0, 
            'return_mean': 154.21621621621622, 
            'return_std': 79.94377589657559, 
            'return_max': 500.0, 
            'len_min': 26, 
            'len_mean': 154.21621621621622, 
            'len_std': 79.94377589657559, 
            'len_max': 500, 
            'monitor_return_min': 26.0, 
            'monitor_return_mean': 154.21621621621622, 
            'monitor_return_std': 79.94377589657559, 
            'monitor_return_max': 500.0
        }
    INFO - eval_policy - Completed after 0:00:12


For the non-expert:

.. code-block:: bash

    python -m imitation.scripts.eval_policy with \
            expert.policy_type=ppo \
            expert.loader_kwargs.path=output/train_rl/CartPole-v1/non_expert/policies/final/model.zip \
            environment.gym_id='CartPole-v1' \
            environment.num_vec=1 \
            logging.log_dir=output/eval_policy/CartPole-v1/non_expert


.. code-block:: bash

    INFO - eval_policy - Result: {
            'n_traj': 355, 
            'monitor_return_len': 355, 
            'return_min': 8.0, 
            'return_mean': 28.92676056338028, 
            'return_std': 15.686012049373561, 
            'return_max': 104.0, 
            'len_min': 8, 
            'len_mean': 28.92676056338028, 
            'len_std': 15.686012049373561, 
            'len_max': 104, 
            'monitor_return_min': 8.0, 
            'monitor_return_mean': 28.92676056338028, 
            'monitor_return_std': 15.686012049373561, 
            'monitor_return_max': 104.0
    }
    INFO - eval_policy - Completed after 0:00:17

This will save the monitor csvs (one for each vectorised env, controlled by environment.num_vec). 
We can load these with ``pandas`` and use the ``imitation.test.reward_improvement``
module to compare the performances of the two policies.

.. code-block:: python

    from imitation.testing.reward_improvement import is_significant_reward_improvement

    expert_monitor = pd.read_csv(
        './output/train_rl/CartPole-v1/expert/monitor/mon000.monitor.csv', 
        skiprows=1
    )
    non_expert_monitor = pd.read_csv(
        './output/train_rl/CartPole-v1/non_expert/monitor/mon000.monitor.csv', 
        skiprows=1
    )
    is_significant_reward_improvement(
        non_expert_monitor['r'], 
        expert_monitor['r'], 
        0.05
    )

.. code-block:: bash

    True    


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
