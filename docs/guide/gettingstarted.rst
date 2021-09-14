===============
Getting Started
===============


CLI Quickstart
==============

We provide several CLI scripts as front-ends to the algorithms implemented in ``imitation``.
These use `Sacred <https://github.com/idsia/sacred>`_ for configuration and replicability.

For information on how to configure Sacred CLI options, see the `Sacred docs <https://sacred.readthedocs.io/en/stable/>`_.

.. code-block:: bash

    # Train PPO agent on cartpole and collect expert demonstrations. Tensorboard logs saved
    # in `quickstart/rl/`
    python -m imitation.scripts.expert_demos with fast cartpole log_dir=quickstart/rl/

    # Train GAIL from demonstrations. Tensorboard logs saved in output/ (default log directory).
    python -m imitation.scripts.train_adversarial with fast gail cartpole \
        rollout_path=quickstart/rl/rollouts/final.pkl

    # Train AIRL from demonstrations. Tensorboard logs saved in output/ (default log directory).
    python -m imitation.scripts.train_adversarial with fast airl cartpole \
        rollout_path=quickstart/rl/rollouts/final.pkl


.. note::
  Remove the ``fast`` option from the commands above to allow training run to completion.

.. tip::
  ``python -m imitation.scripts.expert_demos print_config`` will list Sacred script options.
  These configuration options are also documented in each script's docstrings.


Python Interface Quickstart
===========================

Here's an `example script`_ that loads CartPole demonstrations and trains BC, GAIL, and
AIRL models on that data. You will need to `pip install seals` or `pip install imitation[test]`
to run this.

.. _example script: https://github.com/HumanCompatibleAI/imitation/blob/master/examples/quickstart.py

.. literalinclude :: ../../examples/quickstart.py
   :language: python