.. _gail docs:

================================================
Generative Adversarial Imitation Learning (GAIL)
================================================

`GAIL <https://arxiv.org/abs/1606.03476>`_ learns a policy by simultaneously training it
with a discriminator that aims to distinguish expert trajectories against
trajectories from the learned policy.

Notes
-----
- GAIL paper: `Generative Adversarial Imitation Learning <https://arxiv.org/abs/1606.03476>`_

API
===
.. autoclass:: imitation.algorithms.adversarial.gail.GAIL
    :members:
    :inherited-members:
    :noindex:

.. autoclass:: imitation.algorithms.adversarial.common.AdversarialTrainer
    :members:
    :inherited-members:
    :noindex:
