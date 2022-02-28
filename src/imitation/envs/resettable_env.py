"""Finite-horizon discrete environments with known transition dynamics.

These are handy when you want to perform exact maxent policy optimisation.
"""

import abc
from typing import Optional

import gym
import numpy as np
from gym import spaces
from stable_baselines3.common import vec_env


class ResettableEnv(gym.Env, abc.ABC):
    """ABC for environments that are resettable.

    Specifically, these environments provide oracle access to sample from the initial
    state distribution and transition dynamics, and compute the reward and termination
    condition. Almost all simulated environments can meet these criteria.
    """

    def __init__(self):
        """Builds a ResettableEnv with all attributes initialized to None."""
        self._pomdp_state_space = None
        self._pomdp_observation_space = None
        self._action_space = None
        self.cur_state = None
        self._n_actions_taken = None
        self.rand_state: Optional[np.random.RandomState] = None
        self.seed()

    @abc.abstractmethod
    def initial_state(self):
        """Samples from the initial state distribution."""

    @abc.abstractmethod
    def transition(self, state, action):
        """Samples from transition distribution."""

    @abc.abstractmethod
    def reward(self, state, action, new_state):
        """Computes reward for a given transition."""

    @abc.abstractmethod
    def terminal(self, state, step: int) -> bool:
        """Is the state terminal?"""

    @abc.abstractmethod
    def obs_from_state(self, state):
        """Returns observation produced by a given state."""

    @property
    def pomdp_state_space(self) -> gym.Space:
        """The POMDP's state space.

        In fully observable MDPs, `pomdp_state_space == pomdp_observation_space`.

        Returns:
            The POMDP state space of this environment.
        """
        return self._pomdp_state_space

    @property
    def pomdp_observation_space(self) -> gym.Space:
        """The POMDP's observation space.

        In fully observable MDPs, `pomdp_state_space == pomdp_observation_space`.

        The actual "observation" returned by step() includes both this *and* the state.

        Returns:
            The POMDP observation space of this environment.
        """
        return self._pomdp_observation_space

    @property
    def observation_space(self) -> gym.Space:
        """Combined observation space.

        Dict space, including both the POMDP's state and observation space.
        The intention is to support both algorithms that train on the POMDP's
        observations, as well as allowing some algorithms to "cheat" and
        operate directly on the underlying POMDP states.

        Return type of reset() and component of step().

        Returns:
            The observation space of this environment.
        """
        return gym.spaces.Dict(
            {"obs": self.pomdp_observation_space, "state": self.pomdp_state_space},
        )

    @property
    def action_space(self) -> gym.Space:
        """Action space.

        Parameter type of step().

        Returns:
            The action space of this environment.
        """
        return self._action_space

    @property
    def n_actions_taken(self) -> int:
        """Number of steps taken so far."""
        return self._n_actions_taken

    def seed(self, seed=None):
        if seed is None:
            # Gym API wants list of seeds to be returned for some reason, so
            # generate a seed explicitly in this case
            seed = np.random.randint(0, 1 << 31)
        self.rand_state = np.random.RandomState(seed)
        return [seed]

    def reset(self):
        self.cur_state = self.initial_state()
        self._n_actions_taken = 0
        obs = self.obs_from_state(self.cur_state)
        return {"obs": obs, "state": self.cur_state}

    def step(self, action):
        if self.cur_state is None or self._n_actions_taken is None:
            raise ValueError("Need to call reset() before first step()")

        old_state = self.cur_state
        self.cur_state = self.transition(self.cur_state, action)
        obs = self.obs_from_state(self.cur_state)
        rew = self.reward(old_state, action, self.cur_state)
        self._n_actions_taken += 1
        done = self.terminal(self.cur_state, self._n_actions_taken)

        infos = {"old_state": old_state, "new_state": self.cur_state}
        combined_obs = {"obs": obs, "state": self.cur_state}
        return combined_obs, rew, done, infos


class TabularModelEnv(ResettableEnv, abc.ABC):
    """ABC for tabular environments with known dynamics."""

    def __init__(self):
        """Initialise common attributes of all model-based environments.

        Attributes include current state & number of actions taken so far (initial
        None, so that error can be thrown if reset() is not called), attributes for
        cached observation/action space, and random seed for rollouts.
        """
        super().__init__()

    @property
    def pomdp_state_space(self) -> gym.Space:
        # Construct spaces lazily, so they can depend on properties in subclasses.
        if self._pomdp_state_space is None:
            self._pomdp_state_space = spaces.Discrete(self.state_dim)
        return self._pomdp_state_space

    @property
    def pomdp_observation_space(self) -> gym.Space:
        # Construct spaces lazily, so they can depend on properties in subclasses.
        if self._pomdp_observation_space is None:
            self._pomdp_observation_space = spaces.Box(
                low=float("-inf"),
                high=float("inf"),
                shape=(self.obs_dim,),
                dtype=self.obs_dtype,
            )
        return self._pomdp_observation_space

    @property
    def action_space(self) -> gym.Space:
        # Construct spaces lazily, so they can depend on properties in subclasses.
        if self._action_space is None:
            self._action_space = spaces.Discrete(self.n_actions)
        return self._action_space

    def initial_state(self):
        return self.rand_state.choice(self.n_states, p=self.initial_state_dist)

    def transition(self, state, action):
        out_dist = self.transition_matrix[state, action]
        choice_states = np.arange(self.n_states)
        return int(self.rand_state.choice(choice_states, p=out_dist, size=()))

    def reward(self, state, action, new_state):
        reward = self.reward_matrix[state]
        assert np.isscalar(reward), reward
        return reward

    def terminal(self, state, n_actions_taken):
        return n_actions_taken >= self.horizon

    def obs_from_state(self, state):
        # Copy so it can't be mutated in-place (updates will be reflected in
        # self.observation_matrix!)
        obs = self.observation_matrix[state].copy()
        assert obs.ndim == 1, obs.shape
        return obs

    @property
    def n_states(self):
        """Number of states in this MDP (int)."""
        return self.transition_matrix.shape[0]

    @property
    def n_actions(self):
        """Number of actions in this MDP (int)."""
        return self.transition_matrix.shape[1]

    @property
    def state_dim(self):
        """Size of state vectors for this MDP."""
        return self.observation_matrix.shape[0]

    @property
    def obs_dim(self):
        """Size of observation vectors for this MDP."""
        return self.observation_matrix.shape[1]

    @property
    def obs_dtype(self):
        """Data type of observation vectors (e.g. np.float32)."""
        return self.observation_matrix.dtype

    # ############################### #
    # METHODS THAT MUST BE OVERRIDDEN #
    # ############################### #

    @property
    @abc.abstractmethod
    def transition_matrix(self):
        """3D transition matrix.

        Dimensions correspond to current state, current action, and next state.

        In other words, if `T` is our returned matrix, then `T[s,a,sprime]` is the
        chance of transitioning into state `sprime` after taking action `a` in state
        `s`.
        """

    @property
    @abc.abstractmethod
    def observation_matrix(self):
        """2D observation matrix.

        Dimensions correspond to current state (first dim) and elements of observation
        (second dim).
        """

    @property
    @abc.abstractmethod
    def reward_matrix(self):
        """1D reward matrix with an element corresponding to each state."""

    @property
    @abc.abstractmethod
    def horizon(self):
        """Number of actions that can be taken in an episode."""

    @property
    @abc.abstractmethod
    def initial_state_dist(self):
        """1D vector representing a distribution over initial states."""
        return


class DictExtractWrapper(vec_env.VecEnvWrapper):
    """Extracts key from dict observation of wrapped environment.

    For example, can be used with instances of `ResettableEnv` to extract either
    `'obs'` or `'state'` keys as appropriate. This is useful when you want a model
    to depend on only one, or if the model does not natively support dict-based
    observations.
    """

    def __init__(self, venv: vec_env.VecEnv, key: str):
        """Builds DictExtractWrapper.

        Args:
            venv: A vectorized environment with dict observation space.
            key: The key to extract from observations.

        Raises:
            TypeError: The observation space of `venv` is not a dict.
            KeyError: `key` is not present in the observation space of `venv`.
        """
        if not isinstance(venv.observation_space, gym.spaces.Dict):
            raise TypeError(
                f"Observation space '{venv.observation_space}' is not dict type.",
            )
        if key not in venv.observation_space.spaces:
            raise KeyError(
                f"Unrecognized '{key}'; valid keys = "
                f"{venv.observation_space.spaces.keys()}",
            )
        super().__init__(venv=venv, observation_space=venv.observation_space[key])
        self.key = key

    def reset(self):
        obs = self.venv.reset()
        return obs[self.key]

    def step_wait(self):
        obs, rew, dones, infos = self.venv.step_wait()
        for info in infos:
            if "terminal_observation" in info:
                info["terminal_observation"] = info["terminal_observation"][self.key]
        return obs[self.key], rew, dones, infos
