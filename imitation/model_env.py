"""Finite-horizon discrete environments with known transition dynamics. These
are handy when you want to perform exact maxent policy optimisation."""

import abc

import gym
from gym import spaces
import numpy as np


class ModelBasedEnv(gym.Env, abc.ABC):
    """ABC for tabular environments with known dynamics."""
    def __init__(self):
        self.cur_state = None
        self.n_actions_taken = None
        # we must cache action & observation spaces instead of reconstructing
        # anew so that the random state of the action space (!!) is preserved
        self._action_space = None
        self._observation_space = None
        self.seed()

    @property
    def action_space(self):
        if not self._action_space:
            self._action_space = spaces.Discrete(self.n_actions)
        return self._action_space

    @property
    def observation_space(self):
        if not self._observation_space:
            self._observation_space = spaces.Box(low=float('-inf'),
                                                 high=float('inf'),
                                                 shape=(self.obs_dim, ))
        return self._observation_space

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 1 << 31)
        self.rand_state = np.random.RandomState(seed=seed)
        return [seed]

    def reset(self):
        self.cur_state = 0
        self.n_actions_taken = 0
        # as in step(), we copy so that it can't be mutated in-place (updates
        # will be reflected in self.observation_matrix!)
        return self.observation_matrix[self.cur_state].copy()

    def step(self, action):
        assert self.cur_state is not None \
            and self.n_actions_taken is not None, \
            "remember to call reset() before first step()"
        old_state = self.cur_state
        out_dist = self.transition_matrix[old_state, action]
        choice_states = np.arange(self.n_states)
        next_state = int(
            self.rand_state.choice(choice_states, p=out_dist, size=()))
        self.cur_state = next_state
        self.n_actions_taken += 1
        done = self.n_actions_taken >= self.horizon
        reward = self.reward_matrix[old_state]
        assert np.isscalar(reward), reward
        # copy so that it can't be mutated in-place
        obs = self.observation_matrix[next_state].copy()
        assert obs.ndim == 1, obs.shape
        infos = {"old_state": old_state, "new_state": next_state}
        return obs, reward, done, infos

    @property
    def n_states(self):
        """Number of states in this MDP (int)."""
        return self.transition_matrix.shape[0]

    @property
    def n_actions(self):
        """Number of actions in this MDP (int)."""
        return self.transition_matrix.shape[1]

    @property
    def obs_dim(self):
        """Size of observation vectors for this MDP."""
        return self.observation_matrix.shape[-1]

    # ############################### #
    # METHODS THAT MUST BE OVERRIDDEN #
    # ############################### #

    @property
    @abc.abstractmethod
    def transition_matrix(self):
        """Yield a 3D transition matrix with dimensions corresponding to
        current state, current action, and next state (in that order). In other
        words, if `T` is our returned matrix, then `T[s,a,sprime]` is the
        chance of transitioning into state `sprime` after taking action `a` in
        state `s`."""

    @property
    @abc.abstractmethod
    def observation_matrix(self):
        """Yields 2D observation matrix with dimensions corresponding to
        current state (first dim) and elements of observation (second dim)."""

    @property
    @abc.abstractmethod
    def reward_matrix(self):
        """Yields 2D reward matrix with dimensions corresponding to current
        state and current action."""

    @property
    @abc.abstractmethod
    def horizon(self):
        """Number of actions that can be taken in an episode."""
