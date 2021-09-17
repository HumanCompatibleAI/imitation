"""Generative Adversarial Imitation Learning (GAIL)."""

from typing import Optional

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
        reward_net: reward_nets.RewardNet,
    ):
        """Construct discriminator network.

        Args:
            reward_net: A reward network used to compute the logits for GAIL.
                This is a bit of an abuse of terminology: the actual "reward"
                given by GAIL is the negative logsigmoid of this output. But
                this works as discriminator logits and reward are both scalar outputs.
        """
        super().__init__(reward_net=reward_net)

    def logits_gen_is_high(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """Compute the discriminator's logits for each state-action sample."""
        logits = self.reward_net(state, action, next_state, done)
        assert logits.shape == state.shape[:1]
        return logits

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

    reward_test = reward_train


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
        discrim = DiscrimNetGAIL(reward_net=reward_net)
        super().__init__(
            demonstrations=demonstrations,
            demo_batch_size=demo_batch_size,
            venv=venv,
            discrim_net=discrim,
            gen_algo=gen_algo,
            **kwargs,
        )
