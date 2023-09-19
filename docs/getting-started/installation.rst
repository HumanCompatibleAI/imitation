============
Installation
============

Prerequisites
-------------

- Python 3.8+
- pip (it helps to make sure this is up-to-date: ``pip install -U pip``)
- (on ARM64 Macs) you need to set environment variables due to \
  `a bug in grpcio <https://stackoverflow.com/questions/66640705/how-can-i-install-grpcio-on-an-apple-m1-silicon-laptop>`_:

.. code-block:: bash

    export GRPC_PYTHON_BUILD_SYSTEM_OPENSSL=1
    export GRPC_PYTHON_BUILD_SYSTEM_ZLIB=1

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
