.. imitation documentation master file, created by
   sphinx-quickstart on Wed Jun 26 10:19:27 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Imitation: Clean implementations of Imitation Learning algorithms
=================================================================

GitHub repo: http://github.com/HumanCompatibleAI/imitation


Main Features
~~~~~~~~~~~~~

- Built on and compatible with Stable Baselines 3 (SB3).
- Scripts for training SB3 expert models.
- Data structures and scripts for loading and storing expert demonstrations.
- Implementations of Behavioral Cloning, DAgger, GAIL, and AIRL with customizable
  SB3


.. toctree::
   :maxdepth: 2
   :caption: User Guide
   :hidden:

   guide/install
   guide/gettingstarted


.. toctree::
   :maxdepth: 2
   :caption: Algorithms
   :hidden:

   algorithms/bc
   algorithms/gail
   algorithms/airl
   algorithms/dagger


.. toctree::
   :maxdepth: 3
   :caption: API
   :hidden:

   modules/imitation



Index
==================

* :ref:`genindex`
* :ref:`modindex`
