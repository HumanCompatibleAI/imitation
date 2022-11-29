============
Installation
============

Prerequisites
-------------

- Python 3.8+
- (Optional) OpenGL (to render gym environments)
- (Optional) FFmpeg (to encode videos of renders)
- (Optional) MuJoCo (follow instructions to install `mujoco\_py v1.5 here`_)

.. _mujoco_py v1.5 here:
    https://github.com/openai/mujoco-py/tree/498b451a03fb61e5bdfcb6956d8d7c881b1098b5#install-mujoco


Installation from PyPI
----------------------

To install the latest PyPI release, simply run:

.. code-block:: bash

    pip install imitation


Installation from source
------------------------

Installation from source is useful if you wish to contribute to the development of ``imitation``, or if you need features that have not yet been made available in a stable release:

.. code-block:: bash

    git clone http://github.com/HumanCompatibleAI/imitation
    cd imitation
    pip install -e .

There are also a number of dependencies used for running tests and building the documentation, which can be installed with:

.. code-block:: bash

    pip install -e ".[dev]"
