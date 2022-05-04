"""Common wrapper for adding custom reward values to an environment."""

import collections
import contextlib
import functools
from typing import Deque, Optional

import numpy as np
from stable_baselines3.common import callbacks, vec_env

from imitation.rewards import common


@contextlib.contextmanager
def training_mode(env: vec_env.VecEnvWrapper, mode: bool):
    """Temporarily switch environment wrapper ``env`` to specified training ``mode``.

    Args:
        env: The env to switch the mode of.
        mode: whether to set training mode (``True``) or evaluation (``False``).

    Yields:
        The module `m`.
    """
    if isinstance(env, RewardVecEnvWrapper):
        # Modified from Christoph Heindl's method posted on:
        # https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/3
        old_mode = env.training
        env.train(mode)
        try:
            yield env
        finally:
            env.train(old_mode)
    else:
        # Do nothing if not a RewardVecEnvWrapper
        yield env


training = functools.partial(training_mode, mode=True)
evaluating = functools.partial(training_mode, mode=False)


class WrappedRewardCallback(callbacks.BaseCallback):
    """Logs mean wrapped reward as part of RL (or other) training."""

    def __init__(self, episode_rewards: Deque[float], *args, **kwargs):
        """Builds WrappedRewardCallback.

        Args:
            episode_rewards: A queue that episode rewards will be placed into.
            *args: Passed through to `callbacks.BaseCallback`.
            **kwargs: Passed through to `callbacks.BaseCallback`.
        """
        self.episode_rewards = episode_rewards
        super().__init__(self, *args, **kwargs)

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        if len(self.episode_rewards) == 0:
            return
        mean = sum(self.episode_rewards) / len(self.episode_rewards)
        self.logger.record("rollout/ep_rew_wrapped_mean", mean)


class RewardVecEnvWrapper(vec_env.VecEnvWrapper):
    """Uses a provided reward_fn to replace the reward function returned by `step()`.

    Automatically resets the inner VecEnv upon initialization. A tricky part
    about this class is keeping track of the most recent observation from each
    environment.

    Will also include the previous reward given by the inner VecEnv in the
    returned info dict under the `original_env_rew` key.
    """

    def __init__(
        self,
        venv: vec_env.VecEnv,
        reward_fn: common.RewardFn,
        eval_reward_fn: Optional[common.RewardFn] = None,
        ep_history: int = 100,
        training: bool = False,
    ):
        """Builds RewardVecEnvWrapper.

        Args:
            venv: The VecEnv to wrap.
            reward_fn: A function that wraps takes in vectorized transitions
                (obs, act, next_obs) a vector of episode timesteps, and returns a
                vector of rewards.
            eval_reward_fn: The reward function to use during evaluating mode.
            ep_history: The number of episode rewards to retain for computing
                mean reward.
            training: the mode.
        """
        assert not isinstance(venv, RewardVecEnvWrapper)
        super().__init__(venv)
        self.episode_rewards = collections.deque(maxlen=ep_history)
        self._cumulative_rew = np.zeros((venv.num_envs,))
        self.reward_fn = reward_fn
        self.eval_reward_fn = eval_reward_fn if eval_reward_fn else reward_fn
        self.training = training
        self._old_obs = None
        self._actions = None
        self.reset()

    def make_log_callback(self) -> WrappedRewardCallback:
        """Creates `WrappedRewardCallback` connected to this `RewardVecEnvWrapper`."""
        return WrappedRewardCallback(self.episode_rewards)

    @property
    def envs(self):
        return self.venv.envs

    def reset(self):
        self._old_obs = self.venv.reset()
        return self._old_obs

    def step_async(self, actions):
        self._actions = actions
        return self.venv.step_async(actions)

    def step_wait(self):
        obs, old_rews, dones, infos = self.venv.step_wait()

        # The vecenvs automatically reset the underlying environments once they
        # encounter a `done`, in which case the last observation corresponding to
        # the `done` is dropped. We're going to pull it back out of the info dict!
        obs_fixed = []
        for single_obs, single_done, single_infos in zip(obs, dones, infos):
            if single_done:
                single_obs = single_infos["terminal_observation"]

            obs_fixed.append(single_obs)
        obs_fixed = np.stack(obs_fixed)

        reward_fn = self.reward_fn if self.training else self.eval_reward_fn
        rews = reward_fn(self._old_obs, self._actions, obs_fixed, np.array(dones))
        assert len(rews) == len(obs), "must return one rew for each env"
        done_mask = np.asarray(dones, dtype="bool").reshape((len(dones),))

        # Update statistics
        self._cumulative_rew += rews
        for single_done, single_ep_rew in zip(dones, self._cumulative_rew):
            if single_done:
                self.episode_rewards.append(single_ep_rew)
        self._cumulative_rew[done_mask] = 0

        # we can just use obs instead of obs_fixed because on the next iteration
        # after a reset we DO want to access the first observation of the new
        # trajectory, not the last observation of the old trajectory
        self._old_obs = obs
        for info_dict, old_rew in zip(infos, old_rews):
            info_dict["original_env_rew"] = old_rew
        return obs, rews, dones, infos

    def train(self, mode: bool):
        self.training = mode
