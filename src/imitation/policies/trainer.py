"""Training policies with a specifiable reward function and collect trajectories."""
from typing import List, Union

from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.vec_env import VecEnv

from imitation.data import types
from imitation.data.wrappers import BufferingWrapper
from imitation.rewards.common import RewardFn
from imitation.rewards.reward_nets import RewardNet
from imitation.util.reward_wrapper import RewardVecEnvWrapper


class AgentTrainer:
    """Wrapper for training an SB3 algorithm on an arbitrary reward function.

    TODO(ejnnr): For preference comparisons, we might want to allow something more
    general at some point; only the .train() method is required. Also not clear yet
    how we want to deal with sampling/training: they should probably be separate
    (if only because we need rendering when sampling and using human feedback,
    but not when training), but on the other hand, sampling in addition to training
    can be unnecessary overhead.

    Args:
        algorithm: the stable-baselines algorithm to use for training.
            Its environment must be set.
        reward_fn: either a RewardFn or a RewardNet instance that will supply
            the rewards used for training the agent. In the latter case,
            a simple wrapper is automatically applied to get a RewardFn.
        device: an optional PyTorch device on which the RewardNet expects
            its inputs to be
    """

    def __init__(
        self,
        algorithm: BaseAlgorithm,
        reward_fn: Union[RewardFn, RewardNet],
    ):
        self.algorithm = algorithm
        if isinstance(reward_fn, RewardNet):
            reward_fn = reward_fn.predict
        self.reward_fn = reward_fn

        env = self.algorithm.get_env()
        if not isinstance(env, VecEnv):
            raise ValueError("The environment for the agent algorithm must be set.")
        # The BufferingWrapper records all trajectories, so we can return
        # them after training. This should come first (before the wrapper that
        # changes the reward function), so that we return the original environment
        # rewards.
        env = BufferingWrapper(env)
        env = RewardVecEnvWrapper(env, reward_fn)
        self.algorithm.set_env(env)

    def train(self, total_timesteps: int, **kwargs) -> List[types.TrajectoryWithRew]:
        """Train the agent using the reward function specified during instantiation.

        Args:
            total_timesteps: number of environment timesteps to train for
            **kwargs: other keyword arguments to pass to BaseAlgorithm.train()

        Returns:
            a list of all trajectories that occurred during training, including their
            original environment rewards (rather than the ones computed using reward_fn)
        """
        # to clear the trajectory buffer
        self.venv.reset()
        self.algorithm.learn(total_timesteps=total_timesteps, **kwargs)
        return self._pop_trajectories()

    @property
    def policy(self):
        return self.algorithm.policy

    @property
    def venv(self) -> RewardVecEnvWrapper:
        venv = self.algorithm.get_env()
        assert isinstance(venv, RewardVecEnvWrapper), "RewardVecEnvWrapper missing"
        return venv

    def _pop_trajectories(self) -> List[types.TrajectoryWithRew]:
        # self.venv is the reward wrapper, so self.venv.venv is the BufferingWrapper
        return self.venv.venv.pop_trajectories()
