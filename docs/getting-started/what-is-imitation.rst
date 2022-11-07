==================
What is Imitation?
==================
Imitation is an open-source library providing high-quality, reliable and modular implementations of seven reward and imitation learning algorithms, built on modern backends like PyTorch and Stable Baselines3. It includes three Inverse Reinforcment Learning (IRL) algorithms, three imitation learning algorithms and a preference comparison algorithm. The algorithms follow a consistent interface, making it simple to train and compare a range of algorithms.

A key use case of Imitation is as an experimental baseline. Prior work has shown that
small implementation details in imitation learning algorithms can have significant impacts
on performance `(Orsini et al., 2021) <https://arxiv.org/abs/2106.00672>`_. This could lead to spurious positive results being
reported if a weak experimental baseline were used. To address this challenge, Imitation's algorithms have been carefully benchmarked and compared to prior implementations. It also boasts an extensive test suite that covers over 90% of its code. The codebase is statically type-checked as well.

In addition to providing reliable baselines, Imitation aims to simplify developing novel
reward and Imitation learning algorithms. The implementations are *modular*: users can
freely change the reward or policy network architecture, RL algorithm and optimizer without any changes to the code. Algorithms can be extended by subclassing and overriding the relevant methods. Moreover, to support the develop of entirely novel algorithms, Imitation provides utility methods to handle common tasks such as collecting rollouts.
