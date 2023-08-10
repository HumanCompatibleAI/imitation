"""Interactive policy classes to query humans for actions in simple text-based games."""
import os
from typing import Dict, Optional, Tuple

import numpy as np
from stable_baselines3.common.vec_env import VecEnv

from imitation.data.rollout import generate_trajectories, make_min_episodes
from imitation.policies import base

_WASD_ACTION_MAP: Dict[str, Tuple[int, str]] = {
    "w": (3, "up"),  # first entry is action index, second is action name
    "a": (0, "left"),
    "s": (1, "down"),
    "d": (2, "right"),
}


def _parse_action_map(
    action_map: Dict[str, Tuple[int, str]]
) -> Tuple[Dict[str, int], Dict[str, str]]:
    """Parse action map config into separate action index and action name dicts."""
    action_index = {}
    action_name = {}
    for key, (index, name) in action_map.items():
        action_index[key] = index
        action_name[key] = name
    return action_index, action_name


class TextInteractivePolicy(base.InteractivePolicy):
    """Text-based interactive policy."""

    DEFAULT_ACTION_MAPS = {
        "FrozenLake-v1": _WASD_ACTION_MAP,
        # todo: add other default mappings for other environments
    }

    def __init__(
        self,
        venv: VecEnv,
        action_map: Optional[dict] = None,
        refresh_console: bool = True,  # todo: choose default based on detected env
    ):
        """
        Initialize InteractivePolicy with specified environment and optional action map config.
        The action_map_config argument allows for customization of action input keys.
        """
        super().__init__(venv)
        if not action_map:
            env_id = "FrozenLake-v1"  # todo: attempt to infer from venv
            try:
                action_map = self.DEFAULT_ACTION_MAPS[env_id]
            except KeyError:
                raise ValueError(f"No default action map for the environment {env_id}.")
        try:
            action_map, action_names = _parse_action_map(action_map)
        except TypeError:
            action_names = None  # todo: infer from venv

        self.action_map = action_map
        self.refresh_console = refresh_console

        action_guide = (
            f" ({', '.join(str(k)+': '+str(v) for k, v in action_names.items())})"
            if action_names
            else ""
        )
        self.action_prompt = f"Enter an action{action_guide}: "

    def _refresh_console(self) -> None:
        if os.name == "nt":  # windows
            os.system("cls")
        else:  # unix (including mac)
            os.system("clear")

    def _render(self, obs: np.ndarray) -> None:
        """Print the current state of the environment to the console."""
        if self.refresh_console:
            self._refresh_console()
        self.venv.render(mode="human")

    def _query_action(self, obs: np.ndarray) -> np.ndarray:
        """Query human for an action."""
        while True:
            user_input = input(self.action_prompt).strip().lower()
            if user_input in self.action_map:
                return np.array([self.action_map[user_input]])
            print("Invalid input. Try again.")


if __name__ == "__main__":
    import numpy as np
    from gym.envs.toy_text.frozen_lake import generate_random_map

    from imitation.util.util import make_vec_env

    # todo: also test with gym.env
    env = make_vec_env(
        "FrozenLake-v1",
        rng=np.random.default_rng(),
        n_envs=1,  # easier to play :)
        env_make_kwargs=dict(
            desc=generate_random_map(size=4),
            is_slippery=False,
        ),
    )
    expert = TextInteractivePolicy(env)

    # rollout our interactive expert policy
    generate_trajectories(
        expert,
        env,
        sample_until=make_min_episodes(1),
        rng=np.random.default_rng(),
    )
