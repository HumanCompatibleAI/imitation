"""Soft Q Imitation Learning (SQIL).

Trains a policy via DQN-style Q-learning,
replacing half the buffer with expert demonstrations and adjusting the rewards.
"""
from typing import Any, Dict, Iterable, Optional, Tuple, Type, Union

import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import dqn
from stable_baselines3.common import policies, vec_env
from stable_baselines3.common.buffers import ReplayBuffer
from stable_baselines3.common.type_aliases import (
    MaybeCallback,
    ReplayBufferSamples,
    Schedule,
)
from stable_baselines3.dqn.policies import DQNPolicy

from imitation.algorithms import base as algo_base
from imitation.algorithms.base import AnyTransitions
from imitation.data import types
from imitation.data.rollout import flatten_trajectories
from imitation.data.types import Transitions
from imitation.util import logger as imit_logger
from imitation.util.util import get_first_iter_element


class SQIL(algo_base.DemonstrationAlgorithm):
    """Soft Q Imitation Learning (SQIL).

    Trains a policy via DQN-style Q-learning,
    replacing half the buffer with expert demonstrations and adjusting the rewards.
    """

    expert_buffer: ReplayBuffer

    def __init__(
        self,
        *,
        venv: vec_env.VecEnv,
        demonstrations: Transitions,
        policy: Union[str, Type[DQNPolicy]],
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        learning_rate: Union[float, Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[ReplayBuffer]] = None,
        replay_buffer_kwargs: Optional[Dict[str, Any]] = None,
        optimize_memory_usage: bool = False,
        target_update_interval: int = 10000,
        exploration_fraction: float = 0.1,
        exploration_initial_eps: float = 1.0,
        exploration_final_eps: float = 0.05,
        max_grad_norm: float = 10,
        tensorboard_log: Optional[str] = None,
        policy_kwargs: Optional[Dict[str, Any]] = None,
        verbose: int = 0,
        seed: Optional[int] = None,
        device: Union[th.device, str] = "auto",
        _init_setup_model: bool = True,
    ):
        """Builds SQIL.

        Args:
            venv: The vectorized environment to train on.
            demonstrations: Demonstrations to use for training.
            policy: The policy model to use (SB3).
            custom_logger: Where to log to; if None (default), creates a new logger.
            learning_rate: The learning rate, it can be a function
                of the current progress remaining (from 1 to 0).
            buffer_size: Size of the replay buffer.
            learning_starts: How many steps of the model to collect transitions for
                before learning starts.
            batch_size: Minibatch size for each gradient update.
            tau: The soft update coefficient ("Polyak update", between 0 and 1),
                default 1 for hard update.
            gamma: The discount factor.
            train_freq: Update the model every ``train_freq`` steps. Alternatively
                pass a tuple of frequency and unit
                like ``(5, "step")`` or ``(2, "episode")``.
            gradient_steps: How many gradient steps to do after each
                rollout (see ``train_freq``).
                Set to ``-1`` means to do as many gradient steps as steps done
                in the environment during the rollout.
            replay_buffer_class: Replay buffer class to use
                (for instance ``HerReplayBuffer``).
                If ``None``, it will be automatically selected.
            replay_buffer_kwargs: Keyword arguments to pass
                to the replay buffer on creation.
            optimize_memory_usage: Enable a memory efficient variant of the
                replay buffer at a cost of more complexity.
            target_update_interval: Update the target network every
                ``target_update_interval``  environment steps.
            exploration_fraction: Fraction of entire training period over
                which the exploration rate is reduced.
            exploration_initial_eps: Initial value of random action probability.
            exploration_final_eps: Final value of random action probability.
            max_grad_norm: The maximum value for the gradient clipping.
            tensorboard_log: The log location for tensorboard (if None, no logging).
            policy_kwargs: Additional arguments to be passed to the policy on creation.
            verbose: Verbosity level: 0 for no output, 1 for info messages
                (such as device or wrappers used), 2 for debug messages.
            seed: Seed for the pseudo random generators.
            device: Device (cpu, cuda, ...) on which the code should be run.
                Setting it to auto, the code will be run on the GPU if possible.
            _init_setup_model: Whether or not to build the network
                at the creation of the instance.


        """
        self.venv = venv

        super().__init__(demonstrations=demonstrations, custom_logger=custom_logger)

        self.orig_train_freq = train_freq

        self.dqn = dqn.DQN(
            policy=policy,
            env=venv,
            learning_rate=learning_rate,
            buffer_size=buffer_size,
            learning_starts=learning_starts,
            batch_size=batch_size,
            tau=tau,
            gamma=gamma,
            train_freq=train_freq,
            gradient_steps=gradient_steps,
            replay_buffer_class=replay_buffer_class,
            replay_buffer_kwargs=replay_buffer_kwargs,
            optimize_memory_usage=optimize_memory_usage,
            target_update_interval=target_update_interval,
            exploration_fraction=exploration_fraction,
            exploration_initial_eps=exploration_initial_eps,
            exploration_final_eps=exploration_final_eps,
            max_grad_norm=max_grad_norm,
            tensorboard_log=tensorboard_log,
            policy_kwargs=policy_kwargs,
            verbose=verbose,
            seed=seed,
            device=device,
            _init_setup_model=_init_setup_model,
        )

    def set_demonstrations(self, demonstrations: AnyTransitions) -> None:
        # If demonstrations is a list of trajectories,
        # flatten it into a list of transitions
        if isinstance(demonstrations, Iterable):
            item, demonstrations = get_first_iter_element(  # type: ignore[assignment]
                demonstrations,  # type: ignore[assignment]
            )
            if isinstance(item, types.Trajectory):
                demonstrations = flatten_trajectories(
                    demonstrations,  # type: ignore[arg-type]
                )

        n_samples = len(demonstrations)  # type: ignore[arg-type]
        self.expert_buffer = ReplayBuffer(
            n_samples,
            self.venv.observation_space,
            self.venv.action_space,
            handle_timeout_termination=False,
        )

        for transition in demonstrations:
            self.expert_buffer.add(
                obs=np.array(transition["obs"]),  # type: ignore[index]
                next_obs=np.array(transition["next_obs"]),  # type: ignore[index]
                action=np.array(transition["acts"]),  # type: ignore[index]
                done=np.array(transition["dones"]),  # type: ignore[index]
                reward=np.array(1),
                infos=[{}],
            )

    def train(self, *, total_timesteps: int):
        self.learn_dqn(total_timesteps=total_timesteps)

    @property
    def policy(self) -> policies.BasePolicy:
        assert isinstance(self.dqn.policy, policies.BasePolicy)
        return self.dqn.policy

    def train_dqn(self, gradient_steps: int, batch_size: int = 100) -> None:

        # Needed to make mypy happy, because SB3 typing is shoddy
        assert isinstance(self.dqn.policy, policies.BasePolicy)

        # Switch to train mode (this affects batch norm / dropout)
        self.dqn.policy.set_training_mode(True)
        # Update learning rate according to schedule
        self.dqn._update_learning_rate(self.dqn.policy.optimizer)

        losses = []
        for _ in range(gradient_steps):
            # Sample replay buffer
            new_data = self.dqn.replay_buffer.sample(  # type: ignore[union-attr]
                batch_size // 2,
                env=self.dqn._vec_normalize_env,
            )
            new_data.rewards.zero_()  # Zero out the rewards

            expert_data = self.expert_buffer.sample(
                batch_size // 2,
                env=self.dqn._vec_normalize_env,
            )

            # Concatenate the two batches of data
            replay_data = ReplayBufferSamples(
                *(
                    th.cat((getattr(new_data, name), getattr(expert_data, name)))
                    for name in new_data._fields
                ),
            )

            with th.no_grad():
                # Compute the next Q-values using the target network
                next_q_values = self.dqn.q_net_target(replay_data.next_observations)
                # Follow greedy policy: use the one with the highest value
                next_q_values, _ = next_q_values.max(dim=1)
                # Avoid potential broadcast issue
                next_q_values = next_q_values.reshape(-1, 1)
                # 1-step TD target
                target_q_values = (
                    replay_data.rewards
                    + (1 - replay_data.dones) * self.dqn.gamma * next_q_values
                )

            # Get current Q-values estimates
            current_q_values = self.dqn.q_net(replay_data.observations)

            # Retrieve the q-values for the actions from the replay buffer
            current_q_values = th.gather(
                current_q_values,
                dim=1,
                index=replay_data.actions.long(),
            )

            # Compute Huber loss (less sensitive to outliers)
            loss = F.smooth_l1_loss(current_q_values, target_q_values)
            losses.append(loss.item())

            # Optimize the policy
            self.dqn.policy.optimizer.zero_grad()
            loss.backward()
            # Clip gradient norm
            # For some reason pytype doesn't see nn.utils, so adding a type ignore
            th.nn.utils.clip_grad_norm_(  # type: ignore[module-attr]
                self.dqn.policy.parameters(),
                self.dqn.max_grad_norm,
            )
            self.dqn.policy.optimizer.step()

        # Increase update counter
        self.dqn._n_updates += gradient_steps

        self.dqn.logger.record(
            "train/n_updates",
            self.dqn._n_updates,
            exclude="tensorboard",
        )
        self.dqn.logger.record("train/loss", np.mean(losses))

    def learn_dqn(
        self,
        total_timesteps: int,
        callback: MaybeCallback = None,
        log_interval: int = 4,
        tb_log_name: str = "run",
        reset_num_timesteps: bool = True,
        progress_bar: bool = False,
    ) -> None:

        total_timesteps, callback = self.dqn._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        while self.dqn.num_timesteps < total_timesteps:
            rollout = self.dqn.collect_rollouts(
                self.dqn.env,  # type: ignore[arg-type]  # This is from SB3 code
                train_freq=self.dqn.train_freq,  # type: ignore[arg-type]  # SB3
                action_noise=self.dqn.action_noise,
                callback=callback,
                learning_starts=self.dqn.learning_starts,
                replay_buffer=self.dqn.replay_buffer,  # type: ignore[arg-type]  # SB3
                log_interval=log_interval,
            )

            if rollout.continue_training is False:
                break

            if (
                self.dqn.num_timesteps > 0
                and self.dqn.num_timesteps > self.dqn.learning_starts
            ):
                # If no `gradient_steps` is specified,
                # do as many gradients steps as steps performed during the rollout
                gradient_steps = (
                    self.dqn.gradient_steps
                    if self.dqn.gradient_steps >= 0
                    else rollout.episode_timesteps
                )
                # Special case when the user passes `gradient_steps=0`
                if gradient_steps > 0:
                    self.train_dqn(
                        batch_size=self.dqn.batch_size,
                        gradient_steps=gradient_steps,
                    )

        callback.on_training_end()
