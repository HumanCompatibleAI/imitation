import abc
from typing import Optional, List

import gym
import matplotlib.pyplot as plt
import numpy as np

import imitation.policies.base as base_policies
from imitation.util import util


class DiscreteInteractivePolicy(base_policies.NonTrainablePolicy, abc.ABC):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        action_names: List[str],
        action_keys: List[str],
        clear_screen_on_query: bool = True,
    ):
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
        )

        assert isinstance(action_space, gym.spaces.Discrete)
        assert len(action_names) == len(action_keys) == action_space.n
        # Names and keys should be unique.
        assert len(set(action_names)) == len(set(action_keys)) == action_space.n

        self.action_names = action_names
        self.action_keys = action_keys
        self.action_key_to_index = {k: i for i, k in enumerate(action_keys)}
        self.clear_screen_on_query = clear_screen_on_query

    def _choose_action(self, obs: np.ndarray) -> np.ndarray:
        if self.clear_screen_on_query:
            util.clear_screen()

        context = self._render(obs)
        key = self._get_input_key()
        self._clean_up(context)

        return np.array([self.action_key_to_index[key]])

    def _get_input_key(self) -> str:
        print(
            "Please select an action. Possible choices in [ACTION_NAME:KEY] format:",
            ", ".join(
                [f"{n}:{k}" for n, k in zip(self.action_names, self.action_keys)]
            ),
        )

        key = input("Your choice (enter key):")
        while key not in self.action_keys:
            key = input("Invalid key, please try again! Your choice (enter key):")

        return key

    @abc.abstractmethod
    def _render(self, obs: np.ndarray) -> Optional[object]:
        """Renders an observation, optionally returns a context object for later cleanup."""

    def _clean_up(self, context: object) -> None:
        """Cleans up after the input has been captured, e.g. stops showing the image."""
        pass


class ImageObsDiscreteInteractivePolicy(DiscreteInteractivePolicy):
    def _render(self, obs: np.ndarray) -> plt.Figure:
        img = self._prepare_obs_image(obs)

        fig, ax = plt.subplots()
        ax.imshow(img)
        ax.axis("off")
        fig.show()

        return fig

    def _clean_up(self, context: plt.Figure) -> None:
        plt.close(context)

    def _prepare_obs_image(self, obs: np.ndarray) -> np.ndarray:
        return obs
