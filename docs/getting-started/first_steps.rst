.. _First Steps:

===========
First Steps
===========

Imitation can be used in two main ways: through its command-line interface (CLI) or Python API.
The CLI allows you to quickly train and test algorithms and policies directly from the command line.
The Python API provides greater flexibility and extensibility, and allows you to inter-operate with your existing Python environment.

CLI Quickstart
==============

We provide several CLI scripts as front-ends to the algorithms implemented in ``imitation``.
These use `Sacred <https://github.com/idsia/sacred>`_ for configuration and replicability.

For information on how to configure Sacred CLI options, see the `Sacred docs <https://sacred.readthedocs.io/en/stable/>`_.

.. literalinclude :: ../../examples/quickstart.sh
   :language: bash

.. note::
  Remove the ``fast`` options from the commands above to allow training run to completion.

.. tip::
  ``python -m imitation.scripts.train_rl print_config`` will list Sacred script options.
  These configuration options are also documented in each script's docstrings.


Python Interface Quickstart
===========================

Here's an `example script`_ that loads CartPole demonstrations and trains BC, GAIL, and
AIRL models on that data. You will need to ``pip install seals`` or ``pip install imitation[test]``
to run this.

.. _example script: https://github.com/HumanCompatibleAI/imitation/blob/master/examples/quickstart.py

.. literalinclude :: ../../examples/quickstart.py
   :language: python
