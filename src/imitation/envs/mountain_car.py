import gym
import gym.envs.classic_control
import numpy as np
from seals import util


class ShapedMountainCar(gym.envs.classic_control.MountainCarEnv):
    def __init__(
        self, 
        shaping: str = "unshaped",
        gamma: float = 0.99,
    ):
        super().__init__()
        self.gamma = gamma

        if shaping == "unshaped":
            def potential(obs):
                return 0
        elif shaping == "dense":
            def potential(obs):
                pos, vel = obs
                # simple potential shaping: energy as a proxy for value
                return 0.5 * vel**2 + self.gravity * self._height(pos)
        elif shaping == "antidense":
            def potential(obs):
                pos, vel = obs
                return -0.5 * vel**2 - self.gravity * self._height(pos)
        elif shaping == "random":
            rng = np.random.default_rng(0)
            a, b = 2 * rng.random(2) - 2
            c, d = 2 * np.pi * rng.random(2) - np.pi
            def potential(obs):
                pos, vel = obs
                # just some semi-complex random shaping
                return np.sin(a * pos + b) * np.sin(c * vel + d)
        else:
            raise ValueError(f"Unknow shaping {shaping}")
        
        self._potential = potential

    
    def step(self, action):
        next_obs, rew, done, info = super().step(action)
        rew += self._shaping(self.state, next_obs)
        return next_obs, rew, done, info
    
    def _shaping(self, obs, next_obs):
        current_potential = self._potential(obs)
        # done is always False after wrapping
        next_potential = self._potential(next_obs)
        return self.gamma * next_potential - current_potential

# just a temporary environment so we can pass it to gym.make below
gym.register(
    id="imitation/DummyMountainCar-v0",
    entry_point=ShapedMountainCar,
    max_episode_steps=200,
)

def mountain_car(**kwargs):
    # https://github.com/HumanCompatibleAI/seals/blob/master/src/seals/classic_control.py
    env = util.make_env_no_wrappers("imitation/DummyMountainCar-v0", **kwargs)
    env = util.ObsCastWrapper(env, dtype=np.float32)
    env = util.AbsorbAfterDoneWrapper(env)
    return env

gym.register(
    id="imitation/MountainCar-v0",
    entry_point=mountain_car,
    max_episode_steps=200,
)
