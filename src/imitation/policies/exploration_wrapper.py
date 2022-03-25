"""Wrapper to turn a policy into a more exploratory version."""

from typing import Optional

import numpy as np
from stable_baselines3.common import vec_env

from imitation.data import rollout


class ExplorationWrapper:
    """Wraps a PolicyCallable to create a partially randomized version.

    This wrapper randomly switches between two policies: the wrapped policy,
    and a random one. After each action, the current policy is kept
    with a certain probability. Otherwise, one of these two policies is chosen
    at random (without any dependence on what the current policy is).

    The random policy uses the `action_space.sample()` method.
    """

    def __init__(
        self,
        policy_callable: rollout.PolicyCallable,
        venv: vec_env.VecEnv,
        random_prob: float,
        switch_prob: float,
        seed: Optional[int] = None,
    ):
        """Initializes the ExplorationWrapper.

        Args:
            policy_callable: The policy callable to randomize.
            venv: The environment to use (needed for sampling random actions).
            random_prob: The probability of picking the random policy when switching.
            switch_prob: The probability of switching away from the current policy.
            seed: The random seed to use.
        """
        self.wrapped_policy = policy_callable
        self.random_prob = random_prob
        self.switch_prob = switch_prob
        self.venv = venv

        self.rng = np.random.RandomState(seed)
        self.venv.action_space.seed(seed)

        self.current_policy = policy_callable
        # Choose the initial policy at random
        self._switch()

    def _random_policy(self, obs: np.ndarray) -> np.ndarray:
        acts = [self.venv.action_space.sample() for _ in range(len(obs))]
        return np.stack(acts, axis=0)

    def _switch(self) -> None:
        """Pick a new policy at random."""
        if self.rng.rand() < self.random_prob:
            self.current_policy = self._random_policy
        else:
            self.current_policy = self.wrapped_policy

    def __call__(self, obs: np.ndarray) -> np.ndarray:
        acts = self.current_policy(obs)
        if self.rng.rand() < self.switch_prob:
            self._switch()
        return acts
