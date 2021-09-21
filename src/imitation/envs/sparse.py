"""Sparse versions of Mujoco environments."""

import gym
import numpy as np
from gym.envs.mujoco import reacher


class SparseReacher(reacher.ReacherEnv):
    def step(self, a):
        vec = self.get_body_com("fingertip") - self.get_body_com("target")
        reward = (np.linalg.norm(vec) < 0.01).astype(np.float32)
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        done = False
        return ob, reward, done, {}


gym.register(
    "imitation/SparseReacher-v0",
    entry_point="imitation.envs.sparse:SparseReacher",
)
