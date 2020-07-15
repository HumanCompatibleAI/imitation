import logging
from abc import ABC, abstractmethod
from typing import Callable, Optional

import gym
import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3.common.preprocessing import get_flattened_obs_dim, preprocess_obs
from torch import nn

from imitation.rewards import reward_net
from imitation.util import networks


class DiscrimNet(nn.Module, ABC):
    """Abstract base class for discriminator, used in AIRL and GAIL."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        scale: bool = False,
    ):
        super().__init__()
        self.observation_space = observation_space
        self.action_space = action_space
        self.scale = scale

        # TODO(sam): add back in these histograms, etc. (correct place might be
        # AdversarialTrainer.train_disc_step)
        # tf.summary.histogram("disc_logits", self.disc_logits_gen_is_high)

    @abstractmethod
    def logits_gen_is_high(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """Compute the discriminator's logits for each state-action sample.

        A high value corresponds to predicting generator, and a low value corresponds to
        predicting expert.

        Args:
            state: state at time t.
            action: action taken at time t.
            next_state: state at time t+1.
            done: binary episode completion flag after action at time t.
            log_policy_act_prob: log policy of novice taking `action`. This is
                only used for AIRL.

        Returns:
            disc_logits_gen_is_high: discriminator logits for a sigmoid
                activation. A high output indicates a generator-like transition.
        """

    def disc_loss(self, disc_logits_gen_is_high, labels_gen_is_one) -> th.Tensor:
        """Compute discriminator loss.

        Args:
            disc_logits_gen_is_high: discriminator logits, as produced by
                `logits_gen_is_high`.
            labels_gen_is_one: integer labels, with zero for expert and one for
                generator (novice).

        Returns:
            loss: scalar-valued discriminator loss."""
        return F.binary_cross_entropy_with_logits(
            disc_logits_gen_is_high, labels_gen_is_one.float()
        )

    # TODO(sam): rename these to reward_train() and reward_test() after I've
    # renamed methods below. Also go and rename the RewardNet methods to use
    # the same scheme.
    @abstractmethod
    def reward_test(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        pass

    @abstractmethod
    def reward_train(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        pass

    def device(self) -> th.device:
        """Heuristic to determine which device this module is on."""
        first_param = next(self.parameters())
        return first_param.device

    def predict_reward_train(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:
        """Vectorized reward for training an imitation learning algorithm.

        Args:
            state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            action: The action input. Its shape is
                `(batch_size,) + action_space.shape`. The None dimension is
                expected to be the same as None dimension from `obs_input`.
            next_state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            done: Whether the episode has terminated. Its shape is `(batch_size,)`.
        Returns:
            The rewards. Its shape is `(batch_size,)`.
        """
        return self._eval_reward(
            is_train=True, state=state, action=action, next_state=next_state, done=done
        )

    def predict_reward_test(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:
        """Vectorized reward for training an expert during transfer learning.

        Args:
            state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            act: The action input. Its shape is
                `(batch_size,) + action_space.shape`. The None dimension is
                expected to be the same as None dimension from `obs_input`.
            next_state: The observation input. Its shape is
                `(batch_size,) + observation_space.shape`.
            done: Whether the episode has terminated. Its shape is `(batch_size,)`.
        Returns:
            The rewards. Its shape is `(batch_size,)`.
        """
        return self._eval_reward(
            is_train=False, state=state, action=action, next_state=next_state, done=done
        )

    def _eval_reward(
        self,
        is_train: bool,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ):
        # TODO(sam): the preprocessing code below is identical to
        # RewardNet._eval_reward(). I should rewrite both this method and that
        # one to use the same code underneath.
        dev = self.device()
        state_th = th.as_tensor(state, device=dev)
        action_th = th.as_tensor(action, device=dev)
        next_state_th = th.as_tensor(next_state, device=dev)
        done_th = th.as_tensor(done, device=dev)

        del state, action, next_state, done  # unused

        # preprocess
        state_th = preprocess_obs(state_th, self.observation_space, self.scale)
        action_th = preprocess_obs(action_th, self.action_space, self.scale)
        next_state_th = preprocess_obs(
            next_state_th, self.observation_space, self.scale
        )
        done_th = done_th.to(th.float32)

        n_gen = len(state_th)
        assert state_th.shape == next_state_th.shape
        assert len(action_th) == n_gen

        with th.no_grad():
            if is_train:
                rew_th = self.reward_train(state_th, action_th, next_state_th, done_th)
            else:
                rew_th = self.reward_test(state_th, action_th, next_state_th, done_th)

        rew = rew_th.detach().cpu().numpy().flatten()
        assert rew.shape == (n_gen,)

        return rew


class DiscrimNetAIRL(DiscrimNet):
    r"""The AIRL discriminator for a given RewardNet.

    The AIRL discriminator is of the form
    .. math:: D_{\theta}(s,a) = \frac{\exp(f_{\theta}(s,a)}{\exp(f_{\theta}(s,a) + \pi(a \mid s)}

    where :math:`f_{\theta}` is `self.reward_net`.
    """  # noqa: E501

    def __init__(self, reward_net: reward_net.RewardNet, entropy_weight: float = 1.0):
        """Builds a DiscrimNetAIRL.

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
        self.entropy_weight = entropy_weight
        logging.info("Using AIRL")

    def logits_gen_is_high(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: th.Tensor,
    ) -> th.Tensor:
        """Compute the discriminator's logits for each state-action sample.

        A high value corresponds to predicting generator, and a low value corresponds to
        predicting expert.
        """
        reward_output_train = self.reward_net._reward_train(
            state, action, next_state, done
        )
        return log_policy_act_prob - reward_output_train

    def reward_test(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        rew = self.reward_net._reward_test(state, action, next_state, done)
        assert rew.shape == state.shape[:1]
        return rew

    def reward_train(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
    ) -> th.Tensor:
        """Compute train reward. This reward does *not* include an entropy
        bonus; the entropy bonus should be added directly to PPO, SAC, etc."""
        rew = self.reward_net._reward_train(state, action, next_state, done)
        assert rew.shape == state.shape[:1]
        return rew


DiscrimNetBuilder = Callable[..., nn.Module]
"""Type alias for function that builds a discriminator network.

Takes an observation and action tensor and produces a tuple containing
(1) a list of used TF layers, and (2) output logits.
"""


class ActObsMLP(nn.Module):
    """Simple MLP that takes an action and observation and produces a single
    output."""

    def __init__(
        self, action_space: gym.Space, observation_space: gym.Space, **mlp_kwargs
    ):
        super().__init__()

        in_size = get_flattened_obs_dim(observation_space) + get_flattened_obs_dim(
            action_space
        )
        self.mlp = networks.build_mlp(
            **{"in_size": in_size, "out_size": 1, **mlp_kwargs}
        )

    def forward(self, obs: th.Tensor, acts: th.Tensor) -> th.Tensor:
        cat_inputs = th.cat((obs, acts), dim=1)
        outputs = self.mlp(cat_inputs)
        return outputs.squeeze(1)


class DiscrimNetGAIL(DiscrimNet):
    """The discriminator to use for GAIL."""

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        # FIXME(sam): replace build_discrim_net/build_discrim_net_kwargs with
        # just passing a discrim net straight in
        build_discrim_net: Optional[DiscrimNetBuilder] = None,
        build_discrim_net_kwargs: Optional[dict] = None,
        scale: bool = False,
    ):
        """Construct discriminator network.

        Args:
          observation_space: observation space for this environment.
          action_space: action space for this environment:
          build_discrim_net: a callable that takes an observation input tensor
            and action input tensor as input, then computes the logits
            necessary to feed to GAIL. When called, the function should return
            *both* a `LayersDict` containing all the layers used in
            construction of the discriminator network, and a `th.Tensor`
            representing the desired discriminator logits.
          build_discrim_net_kwargs: optional extra keyword arguments for
            `build_discrim_net()`.
          scale: should inputs be rescaled according to declared observation
            space bounds?
        """
        super().__init__(
            observation_space=observation_space, action_space=action_space, scale=scale
        )

        if build_discrim_net is None:
            if build_discrim_net_kwargs is not None:
                raise ValueError(
                    "must supply build_discrim_net if using " "build_discrim_net_kwargs"
                )
            self.discriminator = ActObsMLP(
                action_space=action_space,
                observation_space=observation_space,
                hid_sizes=(32, 32),
            )
        else:
            if build_discrim_net_kwargs is None:
                raise ValueError(
                    "must supply build_discrim_net_kwargs if " "using build_discrim_net"
                )
            self.discriminator = build_discrim_net(**build_discrim_net_kwargs)

        logging.info("using GAIL")

    def logits_gen_is_high(
        self,
        state: th.Tensor,
        action: th.Tensor,
        next_state: th.Tensor,
        done: th.Tensor,
        log_policy_act_prob: Optional[th.Tensor] = None,
    ) -> th.Tensor:
        """Compute the discriminator's logits for each state-action sample.

        A high value corresponds to predicting generator, and a low value corresponds to
        predicting expert.
        """
        logits = self.discriminator(state, action)
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
