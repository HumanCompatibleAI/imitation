"""Interactive policies that query the user for actions."""

import abc
import collections
import typing

import gym
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3.common import vec_env

import imitation.policies.base as base_policies
from imitation.util import util


class DiscreteInteractivePolicy(base_policies.NonTrainablePolicy, abc.ABC):
    """Abstract class for interactive policies with discrete actions.

    For each query, the observation is rendered and then the action is provided
    as a keyboard input.
    """

    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        action_keys_names: collections.OrderedDict,
        clear_screen_on_query: bool = True,
    ):
        """Builds DiscreteInteractivePolicy.

        Args:
            observation_space: Observation space.
            action_space: Action space.
            action_keys_names: `OrderedDict` containing pairs (key, name) for every
                action, where key will be used in the console interface, and name
                is a semantic action name.
            clear_screen_on_query: If `True`, console will be cleared on every query.
        """
        super().__init__(
            observation_space=observation_space,
            action_space=action_space,
        )

        assert isinstance(action_space, gym.spaces.Discrete)
        assert (
            len(action_keys_names)
            == len(set(action_keys_names.values()))
            == action_space.n
        )

        self.action_keys_names = action_keys_names
        self.action_key_to_index = {
            k: i for i, k in enumerate(action_keys_names.keys())
        }
        self.clear_screen_on_query = clear_screen_on_query

    def _choose_action(self, obs: np.ndarray) -> np.ndarray:
        if self.clear_screen_on_query:
            util.clear_screen()

        context = self._render(obs)
        key = self._get_input_key()
        self._clean_up(context)

        return np.array([self.action_key_to_index[key]])

    def _get_input_key(self) -> str:
        """Obtains input key for action selection."""
        print(
            "Please select an action. Possible choices in [ACTION_NAME:KEY] format:",
            ", ".join([f"{n}:{k}" for k, n in self.action_keys_names.items()]),
        )

        key = input("Your choice (enter key):")
        while key not in self.action_keys_names.keys():
            key = input("Invalid key, please try again! Your choice (enter key):")

        return key

    @abc.abstractmethod
    def _render(self, obs: np.ndarray) -> typing.Optional[object]:
        """Renders an observation, optionally returns a context for later cleanup."""

    def _clean_up(self, context: object) -> None:
        """Cleans up after the input has been captured, e.g. stops showing the image."""
        pass


class ImageObsDiscreteInteractivePolicy(DiscreteInteractivePolicy):
    """DiscreteInteractivePolicy that renders image observations."""

    def _render(self, obs: np.ndarray) -> plt.Figure:
        img = self._prepare_obs_image(obs)

        fig, ax = plt.subplots()
        ax.imshow(img, cmap="gray", vmin=0, vmax=255)  # cmap is ignored for RGB images.
        ax.axis("off")
        fig.show()

        return fig

    def _clean_up(self, context: plt.Figure) -> None:
        plt.close(context)

    def _prepare_obs_image(self, obs: np.ndarray) -> np.ndarray:
        """Applies any required observation processing to get an image to show."""
        return obs


ATARI_ACTION_NAMES_TO_KEYS = {
    "NOOP": "1",
    "FIRE": "2",
    "UP": "w",
    "RIGHT": "d",
    "LEFT": "a",
    "DOWN": "x",
    "UPRIGHT": "e",
    "UPLEFT": "q",
    "DOWNRIGHT": "c",
    "DOWNLEFT": "z",
    "UPFIRE": "t",
    "RIGHTFIRE": "h",
    "LEFTFIRE": "f",
    "DOWNFIRE": "b",
    "UPRIGHTFIRE": "y",
    "UPLEFTFIRE": "r",
    "DOWNRIGHTFIRE": "n",
    "DOWNLEFTFIRE": "v",
}


class AtariInteractivePolicy(ImageObsDiscreteInteractivePolicy):
    """Interactive policy for Atari environments."""

    def __init__(self, env: typing.Union[gym.Env, vec_env.VecEnv], *args, **kwargs):
        """Builds AtariInteractivePolicy."""
        action_names = (
            env.get_action_meanings()
            if isinstance(env, gym.Env)
            else env.env_method("get_action_meanings", indices=[0])[0]
        )
        action_keys_names = collections.OrderedDict(
            [(ATARI_ACTION_NAMES_TO_KEYS[name], name) for name in action_names],
        )
        super().__init__(
            env.observation_space, env.action_space, action_keys_names, *args, **kwargs,
        )
