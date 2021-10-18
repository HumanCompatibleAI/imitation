"""Adversarial Inverse Reinforcement Learning (AIRL)."""

from typing import Optional

import torch as th
from stable_baselines3.common import base_class, vec_env

from imitation.algorithms import base
from imitation.algorithms.adversarial import common
from imitation.rewards import reward_nets


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
        reward_net: Optional[reward_nets.RewardNet] = None,
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
            reward_net: Reward network; used as part of AIRL discriminator. Defaults to
                `reward_nets.BasicShapedRewardNet` when unspecified.
            **kwargs: Passed through to `AdversarialTrainer.__init__`.

        Raises:
            TypeError: If `gen_algo.policy` does not have an `evaluate_actions`
                attribute (present in `ActorCriticPolicy`), needed to compute
                log-probability of actions.
        """
        if reward_net is None:
            reward_net = reward_nets.BasicShapedRewardNet(
                observation_space=venv.observation_space,
                action_space=venv.action_space,
            )
        self._reward_net = reward_net
        super().__init__(
            demonstrations=demonstrations,
            demo_batch_size=demo_batch_size,
            venv=venv,
            gen_algo=gen_algo,
            disc_parameters=self._reward_net.parameters(),
            **kwargs,
        )
        if not hasattr(self.gen_algo.policy, "evaluate_actions"):
            raise TypeError(
                "AIRL needs a stochastic policy to compute the discriminator output.",
            )

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
        reward_output_train = self._reward_net(state, action, next_state, done)
        # In Fu's AIRL paper (https://arxiv.org/pdf/1710.11248.pdf), the
        # discriminator output was given as exp(r_theta(s,a)) /
        # (exp(r_theta(s,a)) + log pi(a|s)), with a high value corresponding to
        # expert and a low value corresponding to generator (the opposite of
        # our convention).
        #
        # Observe that sigmoid(log pi(a|s) - r(s,a)) = exp(log pi(a|s) -
        # r(s,a)) / (1 + exp(log pi(a|s) - r(s,a))). If we multiply through by
        # exp(r(s,a)), we get pi(a|s) / (pi(a|s) + exp(r(s,a))). This is the
        # original AIRL discriminator expression with reversed logits to match
        # our convention of low = expert and high = generator (like GAIL).
        return log_policy_act_prob - reward_output_train

    @property
    def reward_train(self) -> reward_nets.RewardNet:
        return self._reward_net

    @property
    def reward_test(self) -> reward_nets.RewardNet:
        if isinstance(self._reward_net, reward_nets.ShapedRewardNet):
            return self._reward_net.base
        else:
            return self._reward_net
