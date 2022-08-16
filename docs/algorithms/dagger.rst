.. _dagger docs:

=======================
DAgger
=======================

`DAgger <https://arxiv.org/abs/1011.0686>`_ (Dataset Aggregation) iteratively trains a
policy using supervised learning on a dataset of observation-action pairs from expert demonstrations
(like :ref:`behavioral cloning <behavioral cloning docs>`), runs the policy to gather
observations, queries the expert for good actions on those observations, and adds the
newly labeled observations to the dataset. DAgger improves on behavioral cloning by
training on a dataset that better resembles the observations the trained policy is
likely to encounter, but it requires querying the expert online.

Notes
-----
- DAgger paper: `A Reduction of Imitation Learning and Structured Prediction to No-Regret Online Learning <https://arxiv.org/abs/1011.0686>`_

API
===
.. autoclass:: imitation.algorithms.dagger.InteractiveTrajectoryCollector
    :members:
    :inherited-members:
    :noindex:

.. autoclass:: imitation.algorithms.dagger.DAggerTrainer
    :members:
    :inherited-members:
    :noindex:

.. autoclass:: imitation.algorithms.dagger.SimpleDAggerTrainer
    :members:
    :inherited-members:
    :noindex:
