.. _behavioral cloning docs:

=======================
Behavioral Cloning (BC)
=======================

Behavioral cloning directly learns a policy by using supervised learning on
observation-action pairs from expert demonstrations. It is a simple approach to learning
a policy, but the policy often generalizes poorly and does not recover well from errors.

Alternatives to behavioral cloning include :ref:`DAgger <dagger docs>` (similar but gathers
on-policy demonstrations) and :ref:`GAIL <gail docs>`/:ref:`AIRL <airl docs>` (more robust
approaches to learning from demonstrations).

API
===
.. autoclass:: imitation.algorithms.bc.BC
    :members:
    :noindex:
