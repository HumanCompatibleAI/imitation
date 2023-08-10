"""Interactive policy classes to query humans for actions in Atari games."""
from typing import Optional

import gym
import numpy as np
from stable_baselines3.common import type_aliases

from imitation.data.rollout import generate_trajectories, make_min_episodes
from imitation.policies import base, retro_gym_


class AtariInteractivePolicy(base.InteractivePolicy):
    """GUI-based interactive policy for Atari (WIP)."""

    def __init__(
        self,
        venv: type_aliases.GymEnv,
        sync: bool = True,
        tps: int = 60,
        aspect_ratio: Optional[float] = None,
    ):
        """Initialize AtariInteractivePolicy with environment."""
        if not isinstance(venv.unwrapped, gym.Env):
            raise ValueError("venv must be a gym.Env")

        if not hasattr(venv.unwrapped, "buttons"):
            raise ValueError("venv must have a buttons attribute")

        super().__init__(venv)
        self.interactive = retro_gym_.RetroInteractive(  # type: ignore
            env=venv.unwrapped,
            sync=sync,
            tps=tps,
            aspect_ratio=aspect_ratio,
        )

    def _render(self, obs: np.ndarray) -> None:
        # self.interactive.get_image(obs, self.venv)
        self.interactive._draw()

    def _query_action(self, obs: np.ndarray) -> np.ndarray:
        # mimic sync behavior by updating for a fixed time step
        return self.interactive._update(0.1)  # todo: adapt _update to return actions


if __name__ == "__main__":
    # demo of TextInteractivePolicy using Pong
    # note: does not work yet
    import retro  # todo: make retro dependency optional
    from stable_baselines3.common import vec_env

    env = retro.make("Pong-Atari2600")

    expert = AtariInteractivePolicy(env)

    # rollout our interactive expert policy
    trajectories = generate_trajectories(  # todo: generate_trajectories assumes VecEnv
        expert,
        vec_env.DummyVecEnv([lambda: env]),
        sample_until=make_min_episodes(1),
        rng=np.random.default_rng(),
    )
    print(trajectories)
