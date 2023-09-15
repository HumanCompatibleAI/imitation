"""Wrapper to turn a policy into a more exploratory version."""

from typing import Dict, Optional, Tuple, Union

import numpy as np
from stable_baselines3.common import vec_env

from imitation.data import rollout
from imitation.util import util


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
        policy: rollout.AnyPolicy,
        venv: vec_env.VecEnv,
        random_prob: float,
        switch_prob: float,
        rng: np.random.Generator,
        deterministic_policy: bool = False,
    ):
        """Initializes the ExplorationWrapper.

        Args:
            policy: The policy to randomize.
            venv: The environment to use (needed for sampling random actions).
            random_prob: The probability of picking the random policy when switching.
            switch_prob: The probability of switching away from the current policy.
            rng: The random state to use for seeding the environment and for
                switching policies.
            deterministic_policy: Whether to make the policy deterministic when not
                exploring. This must be False when ``policy`` is a ``PolicyCallable``.
        """
        policy_callable = rollout.policy_to_callable(policy, venv, deterministic_policy)
        self.wrapped_policy = policy_callable
        self.random_prob = random_prob
        self.switch_prob = switch_prob
        self.venv = venv

        self.rng = rng
        seed = util.make_seeds(self.rng)
        self.venv.action_space.seed(seed)

        self.current_policy = policy_callable
        # Choose the initial policy at random
        self._switch()

    def _random_policy(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]],
        episode_start: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        del state, episode_start  # Unused
        acts = [self.venv.action_space.sample() for _ in range(len(obs))]
        return np.stack(acts, axis=0), None

    def _switch(self) -> None:
        """Pick a new policy at random."""
        if self.rng.random() < self.random_prob:
            self.current_policy = self._random_policy
        else:
            self.current_policy = self.wrapped_policy

    def __call__(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
        input_state: Optional[Tuple[np.ndarray, ...]],
        episode_start: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        del episode_start  # Unused

        if input_state is not None:
            # This checks that we aren't passed a state
            raise ValueError("Exploration wrapper does not support stateful policies.")

        acts, output_state = self.current_policy(observation, None, None)

        if output_state is not None:
            # This checks that the policy doesn't return a state
            raise ValueError("Exploration wrapper does not support stateful policies.")

        if self.rng.random() < self.switch_prob:
            self._switch()
        return acts, None
