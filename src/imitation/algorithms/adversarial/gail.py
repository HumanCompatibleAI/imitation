"""Generative Adversarial Imitation Learning (GAIL)."""

from typing import Mapping, Optional

import gym
import torch as th
from stable_baselines3.common import base_class, vec_env
from torch import nn
from torch.nn import functional as F

from imitation.algorithms import base
from imitation.algorithms.adversarial import common
from imitation.rewards import reward_nets


class DiscrimNetGAIL(common.DiscrimNet):
    """The discriminator to use for GAIL."""

    discriminator: reward_nets.RewardNet

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        discrim_net: Optional[nn.Module] = None,
        normalize_images: bool = False,
    ):
        """Construct discriminator network.

        Args:
            observation_space: observation space for this environment.
            action_space: action space for this environment:
            discrim_net: a Torch module that takes an observation and action
                tensor as input, then computes the logits for GAIL.
            normalize_images: should image observations be normalized to [0, 1]?
        """
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
            normalize_images=normalize_images,
        )

        if discrim_net is None:
            self.discriminator = reward_nets.BasicRewardNet(
                action_space=action_space,
                observation_space=observation_space,
            )
        else:
            self.discriminator = discrim_net

    def logits_gen_is_high(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """Compute the discriminator's logits for each state-action sample."""
        logits = self.discriminator(state, action, next_state, done)
        return logits

    def reward_test(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        rew = self.reward_train(state, action, next_state, done)
        assert rew.shape == state.shape[:1]
        return rew

    def reward_train(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        logits = self.logits_gen_is_high(state, action, next_state, done)
        rew = -F.logsigmoid(logits)
        assert rew.shape == state.shape[:1]
        return rew


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
        discrim_kwargs: Optional[Mapping] = None,
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
            discrim_kwargs: Passed through to `DiscrimNetGAIL.__init__`.
            **kwargs: Passed through to `AdversarialTrainer.__init__`.
        """
        discrim_kwargs = discrim_kwargs or {}
        discrim = DiscrimNetGAIL(
            venv.observation_space,
            venv.action_space,
            discrim_net=reward_net,
            **discrim_kwargs,
        )
        super().__init__(
            demonstrations=demonstrations,
            demo_batch_size=demo_batch_size,
            venv=venv,
            discrim_net=discrim,
            gen_algo=gen_algo,
            **kwargs,
        )
