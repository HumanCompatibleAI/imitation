"""Training policies with a specifiable reward function and collect trajectories."""
import abc
from typing import List, Union

import numpy as np
from stable_baselines3.common import base_class, vec_env

from imitation.data import types, wrappers
from imitation.rewards import common as rewards_common
from imitation.rewards import reward_nets
from imitation.util import reward_wrapper


class TrajectoryGenerator(abc.ABC):
    @abc.abstractmethod
    def sample(self, num_steps: int) -> List[types.TrajectoryWithRew]:
        """Sample a batch of trajectories.

        Args:
            num_steps: All trajectories taken together should
                have at least this many steps.

        Returns:
            A list of sampled trajectories with rewards (which should
            be the environment rewards, not ones from a reward model).
        """

    def train(self, steps: int, **kwargs):
        """Train an agent if the trajector generator uses one.

        By default, this method does nothing and doesn't need
        to be overridden in subclasses that don't require training.

        Args:
            steps: number of environment steps to train for.
            **kwargs: additional keyword arguments to pass on to
                the training procedure.
        """


class AgentTrainer(TrajectoryGenerator):
    """Wrapper for training an SB3 algorithm on an arbitrary reward function."""

    def __init__(
        self,
        algorithm: base_class.BaseAlgorithm,
        reward_fn: Union[rewards_common.RewardFn, reward_nets.RewardNet],
    ):
        """Initialize the agent trainer.

        Args:
            algorithm: the stable-baselines algorithm to use for training.
                Its environment must be set.
            reward_fn: either a RewardFn or a RewardNet instance that will supply
                the rewards used for training the agent.
        """
        self.algorithm = algorithm
        if isinstance(reward_fn, reward_nets.RewardNet):
            reward_fn = reward_fn.predict
        self.reward_fn = reward_fn

        venv = self.algorithm.get_env()
        if not isinstance(venv, vec_env.VecEnv):
            raise ValueError("The environment for the agent algorithm must be set.")
        # The BufferingWrapper records all trajectories, so we can return
        # them after training. This should come first (before the wrapper that
        # changes the reward function), so that we return the original environment
        # rewards.
        self.buffering_wrapper = wrappers.BufferingWrapper(venv)
        self.venv = reward_wrapper.RewardVecEnvWrapper(
            self.buffering_wrapper, reward_fn
        )
        self.algorithm.set_env(self.venv)

    def train(self, steps: int, **kwargs):
        """Train the agent using the reward function specified during instantiation.

        Args:
            steps: number of environment timesteps to train for
            **kwargs: other keyword arguments to pass to BaseAlgorithm.train()

        Returns:
            a list of all trajectories that occurred during training, including their
            original environment rewards (rather than the ones computed using reward_fn)
        """
        # to clear the trajectory buffer
        self.venv.reset()
        self.algorithm.learn(total_timesteps=steps, **kwargs)

    def sample(self, steps: int) -> List[types.TrajectoryWithRew]:
        trajectories = self._pop_trajectories()
        # We typically have more trajectories than are needed.
        # In that case, we use the final trajectories because
        # they are the ones with the most relevant version of
        # the agent.
        # TODO(ejnnr): should we use random ones instead?

        # The easiest way to do this will be to first invert the
        # list and then just take the first trajectories:
        trajectories = trajectories[::-1]

        # Next, we need the cumulative sum of trajectory lengths
        # to determine how many trajectories to return:
        steps_cumsum = np.cumsum([len(traj) for traj in trajectories])

        # TODO(ejnnr): I think it would be better to sample
        # additional trajectories here if needed.
        avail_steps = steps_cumsum[-1]
        if avail_steps == 0:
            # We have this as a special case to give a better error message
            raise RuntimeError(
                "Trajectory buffer is empty, "
                "run AgentTrainer.train() before .sample()."
            )
        if avail_steps < steps:
            raise RuntimeError(
                f"Requested {steps} environment steps, "
                f"but only {avail_steps} available. "
                "Run AgentTrainer.train() with more steps."
            )

        # Now we find the first index that gives us enough
        # total steps:
        idx = (steps_cumsum >= steps).argmax()
        # we need to include the element at position idx
        trajectories = trajectories[: idx + 1]
        # sanity check
        assert sum(len(traj) for traj in trajectories) >= steps
        return trajectories

    @property
    def policy(self):
        return self.algorithm.policy

    def _pop_trajectories(self) -> List[types.TrajectoryWithRew]:
        return self.buffering_wrapper.pop_trajectories()
