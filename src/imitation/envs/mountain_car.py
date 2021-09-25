import gym
import gym.envs.classic_control
import numpy as np
from seals import util


# Declaring these at top level to allow pickling
def random_potential(self, obs):
    pos, vel = obs
    # just some semi-complex random shaping
    return 100 * np.sin(self.a * pos + self.b) * np.sin(self.c * vel + self.d)

def antidense_potential(self, obs):
    pos, vel = obs
    # return 1e4 * (-0.5 * vel**2 - self.gravity * self._height(pos))
    return -1e2 * pos

def dense_potential(self, obs):
    pos, vel = obs
    # return 1e4 * (0.5 * vel**2 + self.gravity * self._height(pos))
    return 1e2 * pos

def unshaped_potential(self, obs):
    return 0

class ShapedMountainCar(gym.envs.classic_control.MountainCarEnv):
    def __init__(
        self, 
        shaping: str = "unshaped",
        gamma: float = 0.99,
    ):
        super().__init__()
        self.gamma = gamma

        if shaping == "unshaped":
            self._potential = unshaped_potential
        elif shaping == "dense":
            self._potential = dense_potential
        elif shaping == "antidense":
            self._potential = antidense_potential
        elif shaping == "random":
            rng = np.random.default_rng(0)
            self.a, self.b = 2 * rng.random(2) - 2
            self.c, self.d = 2 * np.pi * rng.random(2) - np.pi
            self._potential = random_potential
        else:
            raise ValueError(f"Unknow shaping {shaping}")
        
    
    def step(self, action):
        next_obs, rew, done, info = super().step(action)
        rew += self._shaping(self.state, next_obs)
        return next_obs, rew, done, info
    
    def _shaping(self, obs, next_obs):
        current_potential = self._potential(self, obs)
        # done is always False after wrapping
        next_potential = self._potential(self, next_obs)
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
