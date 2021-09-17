"""Adversarial Inverse Reinforcement Learning (AIRL)."""

from typing import Mapping, Optional, Type

import torch as th
from stable_baselines3.common import base_class, vec_env

from imitation.algorithms import base
from imitation.algorithms.adversarial import common
from imitation.rewards import reward_nets


class DiscrimNetAIRL(common.DiscrimNet):
    r"""The AIRL discriminator for a given RewardNet.

    The AIRL discriminator is of the form
    .. math:: D_{\theta}(s,a) = \frac{\exp(f_{\theta}(s,a)}{\exp(f_{\theta}(s,a) + \pi(a \mid s)}

    where :math:`f_{\theta}` is `self.reward_net`.
    """  # noqa: E501

    def __init__(self, reward_net: reward_nets.RewardNet, entropy_weight: float = 1.0):
        r"""Builds a DiscrimNetAIRL.

        Args:
            reward_net: A RewardNet, used as $f_{\theta}$ in the discriminator.
            entropy_weight: The coefficient for the entropy regularization term.
                To match the AIRL derivation, it should be 1.0.
                However, empirically a lower value sometimes work better.
        """
        super().__init__(
            observation_space=reward_net.observation_space,
            action_space=reward_net.action_space,
        )
        self.reward_net = reward_net
        # if the reward net has potential shaping, we disable that for testing
        if isinstance(reward_net, reward_nets.ShapedRewardNet):
            self.test_reward_net = reward_net.base
        else:
            self.test_reward_net = reward_net
        self.entropy_weight = entropy_weight

    def logits_gen_is_high(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: th.Tensor,
    ) -> th.Tensor:
        """Compute the discriminator's logits for each state-action sample."""
        if log_policy_act_prob is None:
            raise TypeError(
                "Non-None `log_policy_act_prob` is required for this method.",
            )
        reward_output_train = self.reward_net(state, action, next_state, done)
        # In Fu's AIRL paper (https://arxiv.org/pdf/1710.11248.pdf), the
        # discriminator output was given as exp(r_theta(s,a)) /
        # (exp(r_theta(s,a)) - log pi(a|s)), with a high value corresponding to
        # expert and a low value corresponding to generator (the opposite of
        # our convention).
        #
        # Observe that sigmoid(log pi(a|s) - r(s,a)) = exp(log pi(a|s) -
        # r(s,a)) / (1 + exp(log pi(a|s) - r(s,a))). If we multiply through by
        # exp(r(s,a)), we get pi(a|s) / (pi(a|s) + exp(r(s,a))). This is the
        # original AIRL discriminator expression with reversed logits to match
        # our convention of low = expert and high = generator (like GAIL).
        return log_policy_act_prob - reward_output_train

    def reward_test(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        rew = self.test_reward_net(state, action, next_state, done)
        assert rew.shape == state.shape[:1]
        return rew

    def reward_train(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        """Computes train reward.

        Computed reward does *not* include an entropy bonus. Instead, the
        entropy bonus should be added directly to PPO, SAC, etc.

        Args:
            state: Current states of shape `(batch_size,) + state_shape`.
            action: Actions of shape `(batch_size,) + action_shape`.
            next_state: Successor states of shape `(batch_size,) + state_shape`.
            done: End-of-episode (terminal state) indicator of shape `(batch_size,)`.

        Returns:
            Reward of shape `(batch_size,`).
        """
        rew = self.reward_net(state, action, next_state, done)
        assert rew.shape == state.shape[:1]
        return rew


class AIRL(common.AdversarialTrainer):
    """Adversarial Inverse Reinforcement Learning (`AIRL`_).

    .. _AIRL: https://arxiv.org/abs/1710.11248
    """

    def __init__(
        self,
        *,
        demonstrations: base.AnyTransitions,
        demo_batch_size: int,
        venv: vec_env.VecEnv,
        gen_algo: base_class.BaseAlgorithm,
        # FIXME(sam): pass in reward net directly, not via _cls and _kwargs
        reward_net_cls: Type[reward_nets.RewardNet] = reward_nets.BasicShapedRewardNet,
        reward_net_kwargs: Optional[Mapping] = None,
        discrim_kwargs: Optional[Mapping] = None,
        **kwargs,
    ):
        """Builds an AIRL trainer.

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
            reward_net_cls: Reward network constructor. The reward network is part of
                the AIRL discriminator.
            reward_net_kwargs: Optional keyword arguments to use while constructing
                the reward network.
            discrim_kwargs: Optional keyword arguments to use while constructing the
                DiscrimNetAIRL.
            **kwargs: Passed through to `AdversarialTrainer.__init__`.

        Raises:
            TypeError: If `gen_algo.policy` does not have an `evaluate_actions`
                attribute (present in `ActorCriticPolicy`), needed to compute
                log-probability of actions.
        """
        # TODO(shwang): Maybe offer str=>RewardNet conversion like
        #  stable_baselines3 does with policy classes.
        reward_net_kwargs = reward_net_kwargs or {}
        reward_network = reward_net_cls(
            action_space=venv.action_space,
            observation_space=venv.observation_space,
            # pytype is afraid that we'll directly call RewardNet(),
            # which is an abstract class, hence the disable.
            **reward_net_kwargs,  # pytype: disable=not-instantiable
        )

        discrim_kwargs = discrim_kwargs or {}
        discrim = DiscrimNetAIRL(reward_network, **discrim_kwargs)
        super().__init__(
            demonstrations=demonstrations,
            demo_batch_size=demo_batch_size,
            venv=venv,
            discrim_net=discrim,
            gen_algo=gen_algo,
            **kwargs,
        )

        if not hasattr(self.gen_algo.policy, "evaluate_actions"):
            raise TypeError(
                "AIRL needs a stochastic policy to compute the discriminator output.",
            )
