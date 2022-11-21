.. imitation documentation master file, created by
   sphinx-quickstart on Wed Jun 26 10:19:27 2019.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

=========
Imitation
=========

**Imitation provides clean implementations of imitation and reward learning algorithms**, under a unified and user-friendly API.
Currently, we have implementations of Behavioral Cloning, `DAgger <https://arxiv.org/pdf/1011.0686.pdf>`_
(with synthetic examples), density-based reward modeling, `Maximum Causal Entropy Inverse Reinforcement Learning <https://www.cs.cmu.edu/~bziebart/publications/maximum-causal-entropy.pdf>`_,
`Adversarial Inverse Reinforcement Learning <https://arxiv.org/abs/1710.11248>`_,
`Generative Adversarial Imitation Learning <https://arxiv.org/abs/1606.03476>`_, and
`Deep RL from Human Preferences <https://arxiv.org/abs/1706.03741>`_.

You can find us on GitHub at http://github.com/HumanCompatibleAI/imitation.


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
   :caption: Getting Started
   :hidden:

   getting-started/installation
   getting-started/what-is-imitation
   getting-started/variable-horizon
   getting-started/first-steps

.. toctree::
   :maxdepth: 2
   :caption: Algorithms
   :hidden:

   algorithms/bc
   algorithms/gail
   algorithms/airl
   algorithms/dagger
   algorithms/density
   algorithms/mce_irl
   algorithms/preference_comparisons

.. toctree::
   :maxdepth: 2
   :caption: Tutorials
   :hidden:

   tutorials/1_train_bc
   tutorials/2_train_dagger
   tutorials/3_train_gail
   tutorials/4_train_airl
   tutorials/5_train_preference_comparisons
   tutorials/5a_train_preference_comparisons_with_cnn
   tutorials/6_train_mce
   tutorials/7_train_density

.. toctree::
   :maxdepth: 2
   :caption: Experts
   :hidden:

   experts/loading-experts


API Reference
~~~~~~~~~~~~~

.. autosummary::
   :toctree: _api
   :caption: API Reference
   :recursive:
   :template: autosummary/module.rst

   imitation


.. toctree::
   :maxdepth: 1
   :caption: Development
   :hidden:

   development/developer
   development/contributing/index
   development/release-notes
   development/license




Index
==================

* :ref:`genindex`
* :ref:`modindex`
