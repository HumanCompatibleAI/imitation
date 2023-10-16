======================
What is ``imitation``?
======================
``imitation`` is an open-source library providing high-quality, reliable and modular implementations of seven reward and imitation learning algorithms, built on modern backends like `PyTorch <https://pytorch.org/>`_ and `Stable Baselines3 <https://github.com/DLR-RM/stable-baselines3>`_. It includes implementations of :ref:`Behavioral Cloning (BC) <behavioral cloning docs>`, :ref:`DAgger <dagger docs>`, :ref:`Generative Adversarial Imitation Learning (GAIL) <gail docs>`, :ref:`Adversarial Inverse Reinforcement Learning (AIRL) <airl docs>`, :ref:`Reward Learning through Preference Comparisons <preference comparisons docs>`,  :ref:`Maximum Causal Entropy Inverse Reinforcement Learning (MCE IRL) <mce irl docs>`, and :ref:`Density-based reward modeling <density docs>`. The algorithms follow a consistent interface, making it simple to train and compare a range of algorithms.

A key use case of ``imitation`` is as an experimental baseline. Small implementation details in imitation learning algorithms can have significant impacts
on performance, which can lead to spurious positive results if a weak experimental baseline is used. To address this challenge, ``imitation``'s algorithms have been carefully benchmarked and compared to prior implementations. The codebase is statically type-checked and over 90% of it is covered by automated tests.

In addition to providing reliable baselines, ``imitation`` aims to simplify the process of developing novel reward and imitation learning algorithms. Its implementations are *modular*: users can freely change the reward or policy network architecture, RL algorithm and optimizer without touching the codebase itself. Algorithms can be extended by subclassing and overriding relevant methods. ``imitation`` also provides utility methods to handle common tasks to support the development of entirely novel algorithms.

Our goal for ``imitation`` is to make it easier to use, develop, and compare imitation and reward learning algorithms. The library is in active development, and we welcome contributions and feedback.

Check out our recommended
:ref:`First Steps <First Steps>` for an overview of how to use the library. We also have tutorials, such as :doc:`../tutorials/1_train_bc`, that provide detailed examples of specific algorithms. If you are interested in helping develop ``imitation`` then we suggest you refer to the :ref:`Developer Guide <DevGuide>` as well as more specific guidelines for :ref:`Contributing <Contributing>`.
