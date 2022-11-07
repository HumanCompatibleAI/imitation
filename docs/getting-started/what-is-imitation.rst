==================
What is Imitation?
==================
Imitation is an open-source library providing high-quality, reliable and modular implementations of seven reward and imitation learning algorithms. It includes three Inverse Reinforcment Learning (IRL) algorithms, three imitation learning algorithms and a preference comparison algorithm. Crucially, our algorithms follow a consistent interface, making it simple to train and compare a range of algorithms. Furthermore, imitation is built using modern backends such as PyTorch and Stable Baselines3.

A key use case of Imitation is as an experimental baseline. Prior work has shown that
small implementation details in imitation learning algorithms can have significant impacts
on performance `(Orsini et al., 2021) <https://arxiv.org/abs/2106.00672>`_. This could lead to spurious positive results being
reported if a weak experimental baseline were used. To address this challenge, our algorithms have been carefully benchmarked and compared to prior implementations. Our test suite covers greater than 90% of our code, and we also perform static type checking.

In addition to providing reliable baselines, Imitation aims to simplify developing novel
reward and Imitation learning algorithms. Our implementations are *modular*: users can
freely change the reward or policy network architecture, RL algorithm and optimizer without
any changes to the code. Algorithms can be extended by subclassing and overriding the
relevant methods. Moreover, to support the develop of entirely novel algorithms, Imitation
provides utility methods to handle common tasks such as collecting rollouts.

Features
=========

* **Comprehensive**: Imitation implements seven algorithms spanning a range of reward and
  imitation learning styles. Our IRL algorithms consist of 1) the seminal tabular method
  Maximum Causal Entropy IRL (MCE IRL; Ziebart et al., 2010), 2) a baseline based on
  density estimation, and 3) the state-of-the-art approach Adversarial IRL (AIRL; Fu et al.,
  2018). For imitation learning, we include 1) the simple Behavioral Cloning (BC) algorithm,
  1) a variant DAgger (Ross et al., 2011) that learns from interactive demonstrations, and
  2) the state-of-the-art Generative Advesarial Imitation Learning (Ho and Ermon, 2016)
  algorithm. Finally, we also include Deep RL from Human Preferences (DRLHP; Christiano
  et al., 2017) that infers a reward function from comparisons between trajectory fragments.
* **Consistent Interface**: We provide a unified API for all algorithms, inheriting from a common base class
  ``BaseImitationAlgorithm``. Algorithms diverge only where strictly necessary
  (e.g. a different feedback modality). This makes it simple to automatically test a wide range
  of algorithms against a benchmark suite.
* **Experimental Framework**: We provide scripts to train and evaluate the algorithms,
  making it easy to use the library without writing a single line of code. The scripts follow a
  consistent interface, and we include examples to run all algorithms on a suite of commonly
  used environments. To ensure replicable experiments we use Sacred (Greff et al., 2017) for
  configuration and logging.
* **Modularity**: To support the variety of use cases that arise in research, we have designed
  our implementations to be modular and highly configurable. For example, algorithms can
  be configured to use any of the seven Stable Baselines3 RL algorithms (or a custom al-
  gorithm matching this interface). By contrast, prior implementations often implemented
  imitation learning algorithms by subclassing a specific RL algorithm, requiring substantial
  code modification to be ported to new RL algorithms.

  We have also designed the code to be easy to extend in order to implement novel algo-
  rithms. Each algorithm is implemented by a class with instance methods corresponding to
  each logical step of the algorithm. New algorithms can be implemented simply by subclass-
  ing an existing algorithm and overriding a subset of methods. This power is illustrated by
  our implementations of GAIL and AIRL, which both subclass ``AdversarialTrainer``. They
  differ only in the choice of discriminator, with most training logic shared.
* **Documentation**: Imitation comes with extensive documentation available at
  `https://imitation.readthedocs.io/ <https://imitation.readthedocs.io>`_.
* **High-Quality Implemenations**: We take great care to provide reliable implementations
  of algorithms. Our test suite covers 91% of the entire codebase, with coverage rising to over
  98% for the core code (excluding e.g. training scripts). Additionally, we use type annotations
  throughout, and statically check our code using ``pytype``.

  While our thorough testing and code review help avoid bugs, even apparently minor
  implementation details can have significant impacts on algorithm performance (Engstrom
  et al., 2020). We therefore have also benchmarked our algorithms on common environments.
  We find in Table 1 that our algorithms reach expert-level performance on these environments,
  consistent with the results reported in the original papers.
