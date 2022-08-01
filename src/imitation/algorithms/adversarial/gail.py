"""Generative Adversarial Imitation Learning (GAIL)."""

from typing import Optional

import torch as th
from stable_baselines3.common import base_class, vec_env
from torch.nn import functional as F

from imitation.algorithms import base
from imitation.algorithms.adversarial import common
from imitation.rewards import reward_nets


class RewardNetFromDiscriminatorLogit(reward_nets.RewardNet):
    r"""Converts the discriminator logits raw value to a reward signal.

    Wrapper for reward network that takes in the logits of the discriminator
    probability distribution and outputs the corresponding reward for the GAIL
    algorithm.

    Below is the derivation of the transformation that needs to be applied.

    The GAIL paper defines the cost function of the generator as:

    .. math::

        \log{D}

    as shown on line 5 of Algorithm 1. In the paper, :math:`D` is the probability
    distribution learned by the discriminator, where :math:`D(X)=1` if the trajectory
    comes from the generator, and :math:`D(X)=0` if it comes from the expert.
    In this implementation, we have decided to use the opposite convention that
    :math:`D(X)=0` if the trajectory comes from the generator,
    and :math:`D(X)=1` if it comes from the expert. Therefore, the resulting cost
    function is:

    .. math::

        \log{(1-D)}

    Since our algorithm trains using a reward function instead of a loss function, we
    need to invert the sign to get:

    .. math::

        R=-\log{(1-D)}=\log{\frac{1}{1-D}}

    Now, let :math:`L` be the output of our reward net, which gives us the logits of D
    (:math:`L=\operatorname{logit}{D}`). We can write:

    .. math::

        D=\operatorname{sigmoid}{L}=\frac{1}{1+e^{-L}}

    Since :math:`1-\operatorname{sigmoid}{(L)}` is the same as
    :math:`\operatorname{sigmoid}{(-L)}`, we can write:

    .. math::

        R=-\log{\operatorname{sigmoid}{(-L)}}

    which is a non-decreasing map from the logits of D to the reward.
    """

    def __init__(self, base: reward_nets.RewardNet):
        """Builds LogSigmoidRewardNet to wrap `reward_net`."""
        # TODO(adam): make an explicit RewardNetWrapper class?
        super().__init__(
            observation_space=base.observation_space,
            action_space=base.action_space,
            normalize_images=base.normalize_images,
        )
        self.base = base

    def forward(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        logits = self.base.forward(state, action, next_state, done)
        return -F.logsigmoid(-logits)


class GAIL(common.AdversarialTrainer):
    """Generative Adversarial Imitation Learning (`GAIL`_).

    .. _GAIL: https://arxiv.org/abs/1606.03476
    """

    def __init__(
        self,
        *,
        demonstrations: base.AnyTransitions,
        demo_batch_size: int,
        venv: vec_env.VecEnv,
        gen_algo: base_class.BaseAlgorithm,
        reward_net: reward_nets.RewardNet,
        **kwargs,
    ):
        """Generative Adversarial Imitation Learning.

        Args:
            demonstrations: Demonstrations from an expert (optional). Transitions
                expressed directly as a `types.TransitionsMinimal` object, a sequence
                of trajectories, or an iterable of transition batches (mappings from
                keywords to arrays containing observations, etc).
            demo_batch_size: The number of samples in each batch of expert data. The
                discriminator batch size is twice this number because each discriminator
                batch contains a generator sample for every expert sample.
            venv: The vectorized environment to train in.
            gen_algo: The generator RL algorithm that is trained to maximize
                discriminator confusion. Environment and logger will be set to
                `venv` and `custom_logger`.
            reward_net: a Torch module that takes an observation, action and
                next observation tensor as input, then computes the logits.
                Used as the GAIL discriminator.
            **kwargs: Passed through to `AdversarialTrainer.__init__`.
        """
        # Raw self._reward_net is discriminator logits
        reward_net = reward_net.to(gen_algo.device)
        # Process it to produce output suitable for RL training
        # Applies a -log(sigmoid(-logits)) to the logits (see class for explanation)
        self._processed_reward = RewardNetFromDiscriminatorLogit(reward_net)
        super().__init__(
            demonstrations=demonstrations,
            demo_batch_size=demo_batch_size,
            venv=venv,
            gen_algo=gen_algo,
            reward_net=reward_net,
            **kwargs,
        )

    def logits_expert_is_high(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        r"""Compute the discriminator's logits for each state-action sample.

        Args:
            state: The state of the environment at the time of the action.
            action: The action taken by the expert or generator.
            next_state: The state of the environment after the action.
            done: whether a `terminal state` (as defined under the MDP of the task) has
                been reached.
            log_policy_act_prob: The log probability of the action taken by the
                generator, :math:`\log{P(a|s)}`.

        Returns:
            The logits of the discriminator for each state-action sample.
        """
        del log_policy_act_prob
        logits = self._reward_net(state, action, next_state, done)
        assert logits.shape == state.shape[:1]
        return logits

    @property
    def reward_train(self) -> reward_nets.RewardNet:
        return self._processed_reward

    @property
    def reward_test(self) -> reward_nets.RewardNet:
        return self._processed_reward
