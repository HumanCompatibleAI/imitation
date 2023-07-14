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

Citing imitation
~~~~~~~~~~~~~~~~

If you use ``imitation`` in your research project, please cite our paper to help us track our impact and enable readers to more easily replicate your results. You may use the following BibTeX::

    @misc{gleave2022imitation,
      author = {Gleave, Adam and Taufeeque, Mohammad and Rocamonde, Juan and Jenner, Erik and Wang, Steven H. and Toyer, Sam and Ernestus, Maximilian and Belrose, Nora and Emmons, Scott and Russell, Stuart},
      title = {imitation: Clean Imitation Learning Implementations},
      year = {2022},
      howPublished = {arXiv:2211.11972v1 [cs.LG]},
      archivePrefix = {arXiv},
      eprint = {2211.11972},
      primaryClass = {cs.LG},
      url = {https://arxiv.org/abs/2211.11972},
    }

.. toctree::
   :maxdepth: 2
   :caption: Getting Started
   :hidden:

   getting-started/installation
   getting-started/what-is-imitation
   getting-started/variable-horizon
   getting-started/first-steps
   getting-started/cli

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
   tutorials/8_train_custom_env
   tutorials/9_compare_baselines
   tutorials/trajectories
   tutorials/reward_networks

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
