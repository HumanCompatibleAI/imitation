.. _airl docs:

=================================================
Adversarial Inverse Reinforcement Learning (AIRL)
=================================================
`AIRL <https://arxiv.org/abs/1710.11248>`_, similar to :ref:`GAIL <gail docs>`,
adversarially trains a policy against a discriminator that aims to distinguish the expert
demonstrations from the learned policy. Unlike GAIL, AIRL recovers a reward function
that is more generalizable to changes in environment dynamics.

The expert policy must be stochastic.

Notes
-----
- AIRL paper: `Learning Robust Rewards with Adversarial Inverse Reinforcement Learning <https://arxiv.org/abs/1710.11248>`_


API
===
.. autoclass:: imitation.algorithms.adversarial.airl.AIRL
    :members:
    :inherited-members:
    :noindex:

.. autoclass:: imitation.algorithms.adversarial.common.AdversarialTrainer
    :members:
    :noindex:
