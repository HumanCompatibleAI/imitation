
================================================
Limitations on Horizon Length
================================================

.. warning:: Variable Horizon Environments Considered Harmful


Reinforcement learning (RL) algorithms are commonly trained and evaluated in *variable horizon* environments.
In these environments, the episode ends when some termination condition is reached (rather than after a fixed number of steps).
This typically corresponds to success, such as reaching the top of the mountain in ``MountainCar``, or to failure, such as the pole falling down in ``CartPole``.
A variable horizon will tend to speed up RL training, by increasing the proportion of samples where the agent's actions still have a meaningful impact on the reward, pruning out states that are already a foregone conclusion.

However, termination conditions must be carefully hand-designed for each environment.
Their inclusion therefore provides a significant source of information about the reward.
Evaluating reward and imitation learning algorithms in variable-horizon environments can therefore be deeply misleading.
In fact, reward learning in commonly used variable horizon environments such as ``MountainCar`` and ``CartPole`` can be solved by learning a single bit: the sign of the reward.
Of course, an algorithm being able to learn a single bit predicts very little about its performance in real-world tasks, that do not usually come with such an informative termination condition.

To make matters worse, some algorithms have a strong inductive bias towards a particular sign.
Indeed, Figure 5 of `Kostrikov et al (2021)`_ shows that GAIL is able to reach a third of expert performance even without seeing any expert demonstrations.
Consequently, algorithms that happen to have an inductive bias aligned with the task (e.g. positive reward bias for environments where longer episodes are better) may outperform unbiased algorithms on certain environments.
Conversely, algorithms with a misaligned inductive bias will perform worse than an unbiased algorithm.
This may lead to illusory discrepancies between algorithms, or even different implementations of the same algorithm.

`Kostrikov et al (2021)`_ introduces a way to correct for this bias.
However, this does not solve the problem of information leakage.
Rather, it merely ensures that different algorithms are all able to equally exploit the information leak provided by the termination condition.

In light of this issue, we would strongly recommend users evaluate ``imitation`` and other reward or imitation learning algorithms only in fixed-horizon environments.
This is a common, though unfortunately not ubiquitous, practice in reward learning papers.
For example, `Christiano et al (2017)`_ use fixed horizon environments because:

    Removing variable length episodes leaves the agent with only the information encoded in the
    environment itself; human feedback provides its only guidance about what it ought to do.

Many environments, like ``HalfCheetah``, are naturally fixed-horizon.
Moreover, most variable-horizon tasks can be easily converted into fixed-horizon tasks.
Our sister project `seals`_ provides fixed-horizon versions of many commonly used MuJoCo continuous control tasks, as well as mitigating other potential pitfalls in reward learning evaluation.

Given the serious issues with evaluation and training in variable horizon tasks, ``imitation`` will by default throw an error
if training AIRL, GAIL or DRLHP in variable horizon tasks. If you have read this document and understand the problems that
variable horizon tasks can cause but still want to train in variable horizon settings, you can override this safety check
by setting ``allow_variable_horizon=True``. Note this check is not applied for BC or DAgger, which operate on individual
transitions (not episodes) and so cannot exploit the information leak.

Usage with ``allow_variable_horizon=True`` is not officially supported, and we will not optimize ``imitation`` algorithms
to perform well in this situation, as it would not represent real progress. Examples of situations where setting this
flag may nonetheless be appropriate include:

1. Investigating the bias introduced by variable horizon tasks -- e.g. comparing variable to fixed horizon tasks.
2. For unit tests to verify algorithms continue to run on variable horizon environments.
3. Where the termination condition is trivial (e.g. has the robot fallen over?) and the target behaviour is complex
   (e.g. solve a Rubik's cube). In this case, while the termination condition still helps reward and imitation learning,
   the problem remains highly non-trivial even with this information side-channel. However, the existence of this
   side-channel should of course be prominently disclosed.

See this `GitHub issue`_ for further discussion.

.. _Kostrikov et al (2021):
    https://arxiv.org/pdf/1809.02925.pdf#page=8

.. _Christiano et al (2017):
    https://arxiv.org/pdf/1706.03741.pdf#page=14

.. _seals:
    https://github.com/HumanCompatibleAI/seals

.. _GitHub issue:
    https://github.com/HumanCompatibleAI/imitation/issues/324


Non-Support for Infinite Length Horizons
================================================
At the moment, we do not support infinite-length horizons. Many of the imitation algorithms, especially those relying on RL, do not easily port over to infinite-horizon setups. Similarly, much of the logging and reward calculation logic assumes the existence of a finite horizon. Although we may explore workarounds in the future, this is not a feature that we can currently support.
