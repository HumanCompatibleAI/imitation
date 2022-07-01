"""Common wrapper for adding custom reward values to an environment."""

import collections
import functools
from typing import Deque

import numpy as np
from stable_baselines3.common import buffers, callbacks, vec_env

from imitation.rewards import common


def _flatten_buffer_data(arr: np.ndarray) -> np.ndarray:
    """Flatten venv data in the replay buffer.

    Convert shape from [n_steps, n_envs, ...] (when ... is the shape of the features)
    to [n_steps * n_envs, ...] (which maintain the order)

    Args:
        arr: array to flatten

    Returns:
        flattened array
    """
    shape = arr.shape
    if len(shape) < 3:
        shape = shape + (1,)
    return arr.reshape(shape[0] * shape[1], *shape[2:])


class RewardRelabelCallback(callbacks.BaseCallback):
    """Relabel the reward in a replay buffer for an off-policy RL algorithm if any."""

    def __init__(self, reward_fn: common.RewardFn, *args, **kwargs):
        """Builds RewardRelabelCallback.

        Args:
            reward_fn: a RewardFn that will supply the rewards used
                for training the agent.
            *args: Passed through to `callbacks.BaseCallback`.
            **kwargs: Passed through to `callbacks.BaseCallback`.
        """
        super().__init__(self, *args, **kwargs)
        self.reward_fn = reward_fn

    def _on_step(self) -> bool:
        return True

    def _on_training_start(self) -> None:
        replay_buffer = self.model.replay_buffer
        assert isinstance(replay_buffer, buffers.ReplayBuffer)
        pos = replay_buffer.pos
        if replay_buffer.full:
            pos = replay_buffer.buffer_size
        observations = _flatten_buffer_data(replay_buffer.observations[:pos])
        actions = _flatten_buffer_data(replay_buffer.actions[:pos])
        next_observations = _flatten_buffer_data(replay_buffer.next_observations[:pos])
        dones = _flatten_buffer_data(replay_buffer.dones[:pos])

        # relabel the rewards if there are at least 1 transition selected
        if len(observations) > 0:
            rewards = self.reward_fn(observations, actions, next_observations, dones)
            shape = replay_buffer.rewards.shape
            assert len(shape) == 2
            rewards = rewards.reshape(pos, shape[1], *shape[2:])
            replay_buffer.rewards[:pos] = rewards


def create_rew_relabel_callback(reward_fn: common.RewardFn) -> callbacks.BaseCallback:
    """Create a RewardRelabelCallback with update_stats turning off.

    Args:
        reward_fn: a RewardFn that will supply the rewards used
            for training the agent.

    Returns:
        A RewardRelabelCallback.
    """
    # Note(yawen): By default, reward relabeling should not update any auxiliary stat
    # in RewardNet. This would break when reward_fn doesn't have update_stats as an
    # argument. e.g. a hard-coded reward function.
    relabel_reward_fn = functools.partial(reward_fn, update_stats=False)
    reward_relabel_callback = RewardRelabelCallback(
        reward_fn=relabel_reward_fn,
    )
    return reward_relabel_callback


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
        ep_history: int = 100,
    ):
        """Builds RewardVecEnvWrapper.

        Args:
            venv: The VecEnv to wrap.
            reward_fn: A function that wraps takes in vectorized transitions
                (obs, act, next_obs) a vector of episode timesteps, and returns a
                vector of rewards.
            ep_history: The number of episode rewards to retain for computing
                mean reward.
        """
        assert not isinstance(venv, RewardVecEnvWrapper)
        super().__init__(venv)
        self.episode_rewards = collections.deque(maxlen=ep_history)
        self._cumulative_rew = np.zeros((venv.num_envs,))
        self.reward_fn = reward_fn
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

        rews = self.reward_fn(self._old_obs, self._actions, obs_fixed, np.array(dones))
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
