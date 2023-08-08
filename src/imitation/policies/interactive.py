"""Interactive policy classes to query humans for actions and associated utilities."""

import numpy as np
import torch as th
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.vec_env import VecEnv


def query_human() -> np.ndarray:
    """Get action from human.

    0: Move left
    1: Move down
    2: Move right
    3: Move up

    Get arrow keys from human and convert to action.

    Raises:
        ValueError: If invalid action.

    Returns:
        np.array: Action.
    """
    action = None
    while action is None:
        key = input("Enter action: (w/a/s/d) ")
        try:
            action = {
                "w": 3,
                "a": 0,
                "s": 1,
                "d": 2,
            }[key]
        except KeyError:
            raise ValueError("Invalid action.")
    return np.array([action])


class InteractivePolicy(BasePolicy):
    """Interactive policy that queries a human for actions.

    Initialized with a query function that takes an observation and returns an action.
    """

    def __init__(self, venv: VecEnv, render_mode: str = "human"):
        """Builds InteractivePolicy with specified environment."""
        super().__init__(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
        )
        self.venv = venv
        self.render_mode = render_mode  # todo: infer from venv and make configurable

    def _predict(
        self,
        observation: th.Tensor,
        deterministic: bool = False,
    ) -> th.Tensor:
        """Get the action from a human user."""
        self.venv.render(mode=self.render_mode)
        action = query_human()
        return th.tensor(action)


# for the ALE environments see
# https://github.com/mgbellemare/Arcade-Learning-Environment/
# blob/master/docs/gym-interface.md;
# e.g.
#     env = make_vec_env(
#         "ALE/Pong-v5",
#         n_envs=1,
#         rng=np.random.default_rng(42),
#         env_make_kwargs=dict(
#             render_mode="human",
#         ),
#     )


# also see https://github.com/openai/retro/blob/master/retro/examples/interactive.py
