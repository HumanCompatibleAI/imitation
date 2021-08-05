.. imitation documentation master file, created by
   sphinx-quickstart on Wed Jun 26 10:19:27 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=================================================================
Imitation: Clean implementations of Imitation Learning algorithms
=================================================================

``imitation`` is available on GitHub at http://github.com/HumanCompatibleAI/imitation


Main Features
~~~~~~~~~~~~~

- Built on and compatible with Stable Baselines 3 (SB3).
- Modular Pytorch implementations of Behavioral Cloning, DAgger, GAIL, and AIRL that can
  train arbitrary SB3 policies.
- GAIL and AIRL have customizable reward and discriminator networks.
- Scripts to train policies using SB3 and save rollouts from these policies as synthetic "expert" demonstrations.
- Data structures and scripts for loading and storing expert demonstrations.


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
