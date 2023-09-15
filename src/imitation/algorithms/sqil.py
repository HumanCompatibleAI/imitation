"""Soft Q Imitation Learning (SQIL) (https://arxiv.org/abs/1905.11108).

Trains a policy via DQN-style Q-learning,
replacing half the buffer with expert demonstrations and adjusting the rewards.
"""

from typing import Any, Dict, List, Optional, Type, Union

import numpy as np
import torch as th
from gymnasium import spaces
from stable_baselines3 import dqn
from stable_baselines3.common import (
    buffers,
    off_policy_algorithm,
    policies,
    type_aliases,
    vec_env,
)

from imitation.algorithms import base as algo_base
from imitation.data import rollout, types
from imitation.util import logger, util


class SQIL(algo_base.DemonstrationAlgorithm[types.Transitions]):
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
        policy: Union[str, Type[policies.BasePolicy]],
        custom_logger: Optional[logger.HierarchicalLogger] = None,
        rl_algo_class: Type[off_policy_algorithm.OffPolicyAlgorithm] = dqn.DQN,
        rl_kwargs: Optional[Dict[str, Any]] = None,
    ):
        """Builds SQIL.

        Args:
            venv: The vectorized environment to train on.
            demonstrations: Demonstrations to use for training.
            policy: The policy model to use (SB3).
            custom_logger: Where to log to; if None (default), creates a new logger.
            rl_algo_class: Off-policy RL algorithm to use.
            rl_kwargs: Keyword arguments to pass to the RL algorithm constructor.

        Raises:
            ValueError: if `dqn_kwargs` includes a key
                `replay_buffer_class` or `replay_buffer_kwargs`.
        """
        self.venv = venv

        if rl_kwargs is None:
            rl_kwargs = {}
        # SOMEDAY(adam): we could support users specifying their own replay buffer
        # if we made SQILReplayBuffer a more flexible wrapper. Does not seem worth
        # the added complexity until we have a concrete use case, however.
        if "replay_buffer_class" in rl_kwargs:
            raise ValueError(
                "SQIL uses a custom replay buffer: "
                "'replay_buffer_class' not allowed.",
            )
        if "replay_buffer_kwargs" in rl_kwargs:
            raise ValueError(
                "SQIL uses a custom replay buffer: "
                "'replay_buffer_kwargs' not allowed.",
            )

        print("RL algo class: ", rl_algo_class)
        print("Env: ", venv)
        self.rl_algo = rl_algo_class(
            policy=policy,
            env=venv,
            replay_buffer_class=SQILReplayBuffer,
            replay_buffer_kwargs={"demonstrations": demonstrations},
            **rl_kwargs,
        )

        super().__init__(demonstrations=demonstrations, custom_logger=custom_logger)

    def set_demonstrations(self, demonstrations: algo_base.AnyTransitions) -> None:
        assert isinstance(self.rl_algo.replay_buffer, SQILReplayBuffer)
        self.rl_algo.replay_buffer.set_demonstrations(demonstrations)

    def train(self, *, total_timesteps: int, tb_log_name: str = "SQIL", **kwargs: Any):
        self.rl_algo.learn(
            total_timesteps=total_timesteps,
            tb_log_name=tb_log_name,
            **kwargs,
        )

    @property
    def policy(self) -> policies.BasePolicy:
        assert isinstance(self.rl_algo.policy, policies.BasePolicy)
        return self.rl_algo.policy


class SQILReplayBuffer(buffers.ReplayBuffer):
    """A replay buffer that injects 50% expert demonstrations when sampling.

    This buffer is fundamentally the same as ReplayBuffer,
    but it includes an expert demonstration internal buffer.
    When sampling a batch of data, it will be 50/50 expert and collected data.

    It can be used in off-policy algorithms like DQN/SAC/TD3.

    Here it is used as part of SQIL, where it is used to train a DQN.
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
        """Create a SQILReplayBuffer instance.

        Args:
            buffer_size: Max number of elements in the buffer
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
            buffer_size=buffer_size,
            observation_space=observation_space,
            action_space=action_space,
            device=device,
            n_envs=n_envs,
            optimize_memory_usage=optimize_memory_usage,
            handle_timeout_termination=False,
        )

        self.expert_buffer = buffers.ReplayBuffer(
            buffer_size=0,
            observation_space=observation_space,
            action_space=action_space,
        )
        self.set_demonstrations(demonstrations)

    def set_demonstrations(
        self,
        demonstrations: algo_base.AnyTransitions,
    ) -> None:
        """Set the expert demonstrations to be injected when sampling from the buffer.

        Args:
            demonstrations (algo_base.AnyTransitions): Expert demonstrations.

        Raises:
            NotImplementedError: If `demonstrations` is not a transitions object
                or a list of trajectories.
        """
        # If demonstrations is a list of trajectories,
        # flatten it into a list of transitions
        if not isinstance(demonstrations, types.Transitions):
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

        if not isinstance(demonstrations, types.Transitions):
            raise NotImplementedError(
                f"Unsupported demonstrations type: {demonstrations}",
            )

        n_samples = len(demonstrations)
        self.expert_buffer = buffers.ReplayBuffer(
            buffer_size=n_samples,
            observation_space=self.observation_space,
            action_space=self.action_space,
            handle_timeout_termination=False,
        )

        for transition in demonstrations:
            self.expert_buffer.add(
                obs=transition["obs"],
                next_obs=transition["next_obs"],
                action=transition["acts"],
                done=transition["dones"],
                reward=np.array(1.0),
                infos=[{}],
            )

    def add(
        self,
        obs: np.ndarray,
        next_obs: np.ndarray,
        action: np.ndarray,
        reward: np.ndarray,
        done: np.ndarray,
        infos: List[Dict[str, Any]],
    ) -> None:
        super().add(
            obs=obs,
            next_obs=next_obs,
            action=action,
            reward=np.array(0.0),
            done=done,
            infos=infos,
        )

    def sample(
        self,
        batch_size: int,
        env: Optional[vec_env.VecNormalize] = None,
    ) -> buffers.ReplayBufferSamples:
        """Sample a batch of data.

        Half of the batch will be from expert transitions,
        and the other half will be from the learner transitions.

        Args:
            batch_size: Number of elements to sample in total
            env: associated gym VecEnv to normalize the observations/rewards
                when sampling

        Returns:
            A mix of transitions from the expert and from the learner.
        """
        new_sample_size, expert_sample_size = util.split_in_half(batch_size)
        new_sample = super().sample(new_sample_size, env)
        expert_sample = self.expert_buffer.sample(expert_sample_size, env)

        return type_aliases.ReplayBufferSamples(
            *(
                th.cat((getattr(new_sample, name), getattr(expert_sample, name)))
                for name in new_sample._fields
            ),
        )
