import numpy as np
import torch as th
from gym.vector.utils import spaces
from stable_baselines3.common.preprocessing import get_obs_shape

from imitation.policies.replay_buffer_wrapper import ReplayBufferView
from imitation.rewards.reward_function import RewardFn
from imitation.util import util
from imitation.util.networks import RunningNorm


class StateEntropyReward(RewardFn):
    def __init__(self, nearest_neighbor_k: int, observation_space: spaces.Space):
        self.nearest_neighbor_k = nearest_neighbor_k
        # TODO support n_envs > 1
        self.entropy_stats = RunningNorm(1)
        self.observation_space = observation_space
        self.obs_shape = get_obs_shape(observation_space)
        self.replay_buffer_view = ReplayBufferView(
            np.empty(0, dtype=observation_space.dtype), lambda: slice(0)
        )

    def set_replay_buffer(self, replay_buffer: ReplayBufferView):
        self.replay_buffer_view = replay_buffer

    def __call__(
        self,
        state: np.ndarray,
        action: np.ndarray,
        next_state: np.ndarray,
        done: np.ndarray,
    ) -> np.ndarray:
        # TODO: should this work with torch instead of numpy internally?
        #   (The RewardFn protocol requires numpy)

        all_observations = self.replay_buffer_view.observations
        # ReplayBuffer sampling flattens the venv dimension, let's adapt to that
        all_observations = all_observations.reshape(
            (-1, *state.shape[1:])  # TODO #625: fix self.obs_shape
        )
        # TODO #625: deal with the conversion back and forth between np and torch
        entropies = util.compute_state_entropy(
            th.tensor(state),
            th.tensor(all_observations),
            self.nearest_neighbor_k,
        )
        normalized_entropies = self.entropy_stats.forward(entropies)
        return normalized_entropies.numpy()

    def __getstate__(self):
        state = self.__dict__.copy()
        del state["replay_buffer_view"]
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)
        self.replay_buffer_view = ReplayBufferView(
            np.empty(0, self.observation_space.dtype), lambda: slice(0)
        )
