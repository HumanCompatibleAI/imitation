"""Soft Q Imitation Learning (SQIL) (https://arxiv.org/abs/1905.11108).

Trains a policy via DQN-style Q-learning,
replacing half the buffer with expert demonstrations and adjusting the rewards.
"""
from typing import Any, Dict, Iterable, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from gym import spaces
from stable_baselines3 import dqn
from stable_baselines3.common import buffers, policies, type_aliases, vec_env
from stable_baselines3.dqn.policies import DQNPolicy

from imitation.algorithms import base as algo_base
from imitation.data import rollout, types
from imitation.util import logger, util


class SQIL(algo_base.DemonstrationAlgorithm):
    """Soft Q Imitation Learning (SQIL).

    Trains a policy via DQN-style Q-learning,
    replacing half the buffer with expert demonstrations and adjusting the rewards.
    """

    expert_buffer: buffers.ReplayBuffer

    def __init__(
        self,
        *,
        venv: vec_env.VecEnv,
        demonstrations: Optional[algo_base.AnyTransitions],
        policy: Union[str, Type[DQNPolicy]],
        custom_logger: Optional[logger.HierarchicalLogger] = None,
        learning_rate: Union[float, type_aliases.Schedule] = 1e-4,
        buffer_size: int = 1_000_000,  # 1e6
        learning_starts: int = 50000,
        batch_size: int = 32,
        tau: float = 1.0,
        gamma: float = 0.99,
        train_freq: Union[int, Tuple[int, str]] = 4,
        gradient_steps: int = 1,
        replay_buffer_class: Optional[Type[buffers.ReplayBuffer]] = None,
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
            replay_buffer_class=SQILReplayBuffer,
            replay_buffer_kwargs={"demonstrations": demonstrations},
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

        super().__init__(demonstrations=demonstrations, custom_logger=custom_logger)

    def set_demonstrations(self, demonstrations: algo_base.AnyTransitions) -> None:
        assert isinstance(self.dqn.replay_buffer, SQILReplayBuffer)
        self.dqn.replay_buffer.set_demonstrations(demonstrations)

    def train(self, *, total_timesteps: int):
        self.dqn.learn(total_timesteps=total_timesteps)

    @property
    def policy(self) -> policies.BasePolicy:
        assert isinstance(self.dqn.policy, policies.BasePolicy)
        return self.dqn.policy


class SQILReplayBuffer(buffers.ReplayBuffer):
    """Replay buffer used in off-policy algorithms like SAC/TD3.

    :param buffer_size: Max number of element in the buffer
    :param observation_space: Observation space
    :param action_space: Action space
    :param device: PyTorch device
    :param n_envs: Number of parallel environments
    :param optimize_memory_usage: Enable a memory efficient variant
        of the replay buffer which reduces by almost a factor two the memory used,
        at a cost of more complexity.
        See https://github.com/DLR-RM/stable-baselines3/issues/37#issuecomment-637501195
        and https://github.com/DLR-RM/stable-baselines3/pull/28#issuecomment-637559274
        Cannot be used in combination with handle_timeout_termination.
    :param handle_timeout_termination: Handle timeout termination (due to timelimit)
        separately and treat the task as infinite horizon task.
        https://github.com/DLR-RM/stable-baselines3/issues/284
    """

    def __init__(
        self,
        buffer_size: int,
        observation_space: spaces.Space,
        action_space: spaces.Space,
        demonstrations: algo_base.AnyTransitions,
        device: Union[th.device, str] = "auto",
        n_envs: int = 1,
        optimize_memory_usage: bool = False,
    ):
        """A modification of the SB3 ReplayBuffer.

        This buffer is fundamentally the same as ReplayBuffer,
        but it includes an expert demonstration internal buffer.
        When sampling a batch of data, it will be 50/50 expert and collected data.

        Args:
            buffer_size: Max number of element in the buffer
            observation_space: Observation space
            action_space: Action space
            demonstrations: Expert demonstrations.
            device: PyTorch device.
            n_envs: Number of parallel environments. Defaults to 1.
            optimize_memory_usage: Enable a memory efficient variant
                of the replay buffer which reduces by almost a factor two
                the memory used, at a cost of more complexity.
        """
        super().__init__(
            buffer_size,
            observation_space,
            action_space,
            device,
            n_envs,
            optimize_memory_usage,
            handle_timeout_termination=False,
        )

        self.expert_buffer = self.set_demonstrations(demonstrations)

    def set_demonstrations(
        self,
        demonstrations: algo_base.AnyTransitions,
    ) -> buffers.ReplayBuffer:
        """Set the demonstrations to be used in the buffer.

        Args:
            demonstrations (algo_base.AnyTransitions): Expert demonstrations.

        Returns:
            buffers.ReplayBuffer: The buffer with demonstrations added
        """
        # If demonstrations is a list of trajectories,
        # flatten it into a list of transitions
        if isinstance(demonstrations, Iterable):
            (
                item,
                demonstrations,
            ) = util.get_first_iter_element(  # type: ignore[assignment]
                demonstrations,  # type: ignore[assignment]
            )
            if isinstance(item, types.Trajectory):
                demonstrations = rollout.flatten_trajectories(
                    demonstrations,  # type: ignore[arg-type]
                )

        n_samples = len(demonstrations)  # type: ignore[arg-type]
        expert_buffer = buffers.ReplayBuffer(
            n_samples,
            self.observation_space,
            self.action_space,
            handle_timeout_termination=False,
        )

        for transition in demonstrations:
            expert_buffer.add(
                obs=np.array(transition["obs"]),  # type: ignore[index]
                next_obs=np.array(transition["next_obs"]),  # type: ignore[index]
                action=np.array(transition["acts"]),  # type: ignore[index]
                done=np.array(transition["dones"]),  # type: ignore[index]
                reward=np.array(1),
                infos=[{}],
            )

        return expert_buffer

    def sample(
        self,
        batch_size: int,
        env: Optional[vec_env.VecNormalize] = None,
    ) -> buffers.ReplayBufferSamples:
        """Sample a batch of data.

        Half of the batch will be from the expert buffer,
        and the other half will be from the collected data.

        Args:
            batch_size: Number of element to sample in total
            env: associated gym VecEnv to normalize the observations/rewards
                when sampling

        Returns:
            A batch of samples for DQN

        """
        new_sample_size, expert_sample_size = util.split_in_half(batch_size)

        new_sample = super().sample(new_sample_size, env)
        new_sample.rewards.fill_(0)

        expert_sample = self.expert_buffer.sample(expert_sample_size, env)
        expert_sample.rewards.fill_(1)

        replay_data = type_aliases.ReplayBufferSamples(
            *(
                th.cat((getattr(new_sample, name), getattr(expert_sample, name)))
                for name in new_sample._fields
            ),
        )

        return replay_data
