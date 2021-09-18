"""Generative Adversarial Imitation Learning (GAIL)."""

from typing import Optional

import torch as th
from stable_baselines3.common import base_class, vec_env
from torch import nn
from torch.nn import functional as F

from imitation.algorithms import base
from imitation.algorithms.adversarial import common
from imitation.rewards import reward_nets


class LogSigmoidRewardNet(reward_nets.RewardNet):
    """Wrapper for reward network that takes log sigmoid of wrapped network."""

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
        """Computes negative log sigmoid of base reward network."""
        logits = self.base.forward(state, action, next_state, done)
        return -F.logsigmoid(logits)


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
        reward_net: Optional[nn.Module] = None,
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
            reward_net: a Torch module that takes an observation and action
                tensor as input, then computes the logits for GAIL.
            **kwargs: Passed through to `AdversarialTrainer.__init__`.
        """
        if reward_net is None:
            reward_net = reward_nets.BasicRewardNet(
                observation_space=venv.observation_space,
                action_space=venv.action_space,
            )
        self._discriminator = reward_net.to(gen_algo.device)
        self._reward_net = LogSigmoidRewardNet(self._discriminator)
        super().__init__(
            demonstrations=demonstrations,
            demo_batch_size=demo_batch_size,
            venv=venv,
            gen_algo=gen_algo,
            disc_parameters=self._discriminator.parameters(),
            **kwargs,
        )

    def logits_gen_is_high(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """Compute the discriminator's logits for each state-action sample."""
        logits = self._discriminator(state, action, next_state, done)
        assert logits.shape == state.shape[:1]
        return logits

    @property
    def reward_train(self) -> reward_nets.RewardNet:
        return self._reward_net

    @property
    def reward_test(self) -> reward_nets.RewardNet:
        return self._reward_net
