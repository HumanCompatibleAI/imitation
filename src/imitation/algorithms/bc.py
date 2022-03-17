import itertools
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Type, Union

import gym
import torch as th
import numpy as np
from stable_baselines3.common import utils, vec_env
from stable_baselines3.common.policies import ActorCriticPolicy

from imitation.algorithms.base import (
    AnyTransitions,
    DemonstrationAlgorithm,
    TransitionMapping,
    make_data_loader,
)
from imitation.data import rollout, types
from imitation.policies.base import FeedForward32Policy
from imitation.util import logger


@dataclass(frozen=True)
class BCLoss:
    neglogp: th.Tensor
    entropy: th.Tensor
    ent_loss: th.Tensor
    prob_true_act: th.Tensor
    l2_norm: th.Tensor
    l2_loss: th.Tensor
    loss: th.Tensor


@dataclass(frozen=True)
class BCLossCalculator:
    ent_weight: float
    l2_weight: float

    def __call__(
        self,
        policy: ActorCriticPolicy,
        obs: Union[th.Tensor, np.ndarray],
        acts: Union[th.Tensor, np.ndarray],
    ) -> BCLoss:
        """Calculate the supervised learning loss used to train the behavioral clone.

        Args:
            policy: The actor-critic policy of which to compute the loss.
            obs: The observations seen by the expert.
            acts: The actions taken by the expert.

        Returns:
            loss: The supervised learning loss for the behavioral clone to optimize.
            stats_dict: Statistics about the learning process to be logged.

        """
        _, log_prob, entropy = policy.evaluate_actions(obs, acts)
        prob_true_act = th.exp(log_prob).mean()
        log_prob = log_prob.mean()
        entropy = entropy.mean()

        l2_norms = [th.sum(th.square(w)) for w in policy.parameters()]
        l2_norm = sum(l2_norms) / 2  # divide by 2 to cancel with gradient of square

        ent_loss = -self.ent_weight * entropy
        neglogp = -log_prob
        l2_loss = self.l2_weight * l2_norm
        loss = neglogp + ent_loss + l2_loss

        return BCLoss(neglogp, entropy, ent_loss, prob_true_act, l2_norm, l2_loss, loss)


@dataclass(frozen=True)
class BCTrainer:
    loss: BCLossCalculator
    optimizer: th.optim.Optimizer
    policy: ActorCriticPolicy
    device: th.device  # TODO(max): not sure whether the device belongs in this class

    def __call__(self, demonstration_data: Iterable[TransitionMapping]):
        for batch in demonstration_data:
            obs = th.as_tensor(batch["obs"], device=self.device).detach()
            acts = th.as_tensor(batch["acts"], device=self.device).detach()
            l = self.loss(self.policy, obs, acts)

            self.optimizer.zero_grad()
            l.loss.backward()
            self.optimizer.step()

            yield self.policy, l, batch


def reconstruct_policy(
    policy_path: str,
    device: Union[th.device, str] = "auto",
) -> ActorCriticPolicy:
    """Reconstruct a saved policy.

    Args:
        policy_path: path where `.save_policy()` has been run.
        device: device on which to load the policy.

    Returns:
        policy: policy with reloaded weights.
    """
    policy = th.load(policy_path, map_location=utils.get_device(device))
    assert isinstance(policy, ActorCriticPolicy)
    return policy


class BC(DemonstrationAlgorithm):
    def __init__(
        self,
        *,
        observation_space: gym.Space,
        action_space: gym.Space,
        policy: Optional[ActorCriticPolicy] = None,
        demonstrations: Optional[AnyTransitions] = None,
        batch_size: int = 32,
        optimizer_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        ent_weight: float = 1e-3,
        l2_weight: float = 0.0,
        device: Union[str, th.device] = "auto",
        custom_logger: Optional[logger.HierarchicalLogger] = None,
    ):

        self._demo_data_loader: Optional[Iterable[TransitionMapping]] = None
        self.batch_size = batch_size
        super().__init__(
            demonstrations=demonstrations,
            custom_logger=custom_logger,
        )
        self.tensorboard_step = 0

        self.action_space = action_space
        self.observation_space = observation_space
        self.device = utils.get_device(device)

        if policy is None:
            policy = FeedForward32Policy(
                observation_space=observation_space,
                action_space=action_space,
                # Set lr_schedule to max value to force error if policy.optimizer
                # is used by mistake (should use self.optimizer instead).
                lr_schedule=lambda _: th.finfo(th.float32).max,
            )
        self._policy = policy.to(self.device)
        # TODO(adam): make policy mandatory and delete observation/action space params?
        assert self.policy.observation_space == self.observation_space
        assert self.policy.action_space == self.action_space

        if optimizer_kwargs:
            if "weight_decay" in optimizer_kwargs:
                raise ValueError("Use the parameter l2_weight instead of weight_decay.")
        optimizer_kwargs = optimizer_kwargs or {}
        optimizer = optimizer_cls(
            self.policy.parameters(),
            **optimizer_kwargs,
        )
        loss_computer = BCLossCalculator(ent_weight, l2_weight)
        self.trainer = BCTrainer(loss_computer, optimizer, policy, self.device)

    @property
    def policy(self) -> ActorCriticPolicy:
        return self._policy

    def set_demonstrations(self, demonstrations: AnyTransitions) -> None:
        self._demo_data_loader = make_data_loader(
            demonstrations,
            self.batch_size,
        )

    def train(
        self,
        *,
        n_epochs: Optional[int] = None,
        n_batches: Optional[int] = None,
        on_epoch_end: Optional[Callable[[], None]] = None,
        on_batch_end: Optional[Callable[[], None]] = None,
        log_interval: int = 500,
        log_rollouts_venv: Optional[vec_env.VecEnv] = None,
        log_rollouts_n_episodes: int = 5,
        progress_bar: bool = True,
        reset_tensorboard: bool = False,
    ):
        """Train with supervised learning for some number of epochs.

        Here an 'epoch' is just a complete pass through the expert data loader,
        as set by `self.set_expert_data_loader()`.

        Args:
            n_epochs: Number of complete passes made through expert data before ending
                training. Provide exactly one of `n_epochs` and `n_batches`.
            n_batches: Number of batches loaded from dataset before ending training.
                Provide exactly one of `n_epochs` and `n_batches`.
            on_epoch_end: Optional callback with no parameters to run at the end of each
                epoch.
            on_batch_end: Optional callback with no parameters to run at the end of each
                batch.
            log_interval: Log stats after every log_interval batches.
            log_rollouts_venv: If not None, then this VecEnv (whose observation and
                actions spaces must match `self.observation_space` and
                `self.action_space`) is used to generate rollout stats, including
                average return and average episode length. If None, then no rollouts
                are generated.
            log_rollouts_n_episodes: Number of rollouts to generate when calculating
                rollout stats. Non-positive number disables rollouts.
            progress_bar: If True, then show a progress bar during training.
            reset_tensorboard: If True, then start plotting to Tensorboard from x=0
                even if `.train()` logged to Tensorboard previously. Has no practical
                effect if `.train()` is being called for the first time.
        """

        if reset_tensorboard:
            self.tensorboard_step = 0

        epochs_and_batches_specified = n_epochs is not None and n_batches is not None
        neither_epochs_nor_batches_specified = n_epochs is None and n_batches is None
        if epochs_and_batches_specified or neither_epochs_nor_batches_specified:
            raise ValueError(
                "Must provide exactly one of `n_epochs` and `n_batches` arguments.",
            )

        def demo_data_iterator():
            batch_num = 0
            num_samples_so_far = 0

            for epoch_num in itertools.islice(itertools.count(), n_epochs):
                num_batches_in_epoch = 0
                for batch in self._demo_data_loader:
                    yield batch

                    self.logger.record("bc/batch", batch_num)
                    self.logger.record("bc/samples_so_far", num_samples_so_far)
                    self.logger.record("bc/epoch", epoch_num)

                    batch_num += 1
                    num_batches_in_epoch += 1
                    num_samples_so_far += len(batch["obs"])
                    if on_batch_end is not None:
                        on_batch_end()

                if num_batches_in_epoch == 0:
                    raise AssertionError(
                        f"Data loader returned no data after "
                        f"{batch_num} batches, during epoch "
                        f"{epoch_num} -- did it reset correctly?",
                    )
                if on_epoch_end is not None:
                    on_epoch_end()

        training_data = itertools.islice(demo_data_iterator(), n_batches)

        for batch_num, (policy, loss, batch) in enumerate(self.trainer(training_data)):
            if batch_num % log_interval == 0:
                self._log_dict(loss.__dict__)
                # TODO(shwang): Maybe instead use a callback that can be shared between
                #   all algorithms' `.train()` for generating rollout stats.
                #   EvalCallback could be a good fit:
                #   https://stable-baselines3.readthedocs.io/en/master/guide/callbacks.html#evalcallback
                if log_rollouts_venv is not None and log_rollouts_n_episodes > 0:
                    self.logger.record("batch_size", len(batch["obs"]))
                    self._log_rollout_stats(log_rollouts_venv, log_rollouts_n_episodes)
                self.logger.dump(self.tensorboard_step)
            self.tensorboard_step += 1

    def _log_dict(self, d):
        for k, v in d.items():
            self.logger.record(f"bc/{k}", v)

    def _log_rollout_stats(self, log_rollouts_venv, num_rollout_episodes):
        trajs = rollout.generate_trajectories(
            self.policy,
            log_rollouts_venv,
            rollout.make_min_episodes(num_rollout_episodes),
        )
        stats = rollout.rollout_stats(trajs)

        for k, v in stats.items():
            if "return" in k and "monitor" not in k:
                self.logger.record("rollout/" + k, v)

    def save_policy(self, policy_path: types.AnyPath) -> None:
        """Save policy to a path. Can be reloaded by `.reconstruct_policy()`.

        Args:
            policy_path: path to save policy to.
        """
        th.save(self.policy, policy_path)
