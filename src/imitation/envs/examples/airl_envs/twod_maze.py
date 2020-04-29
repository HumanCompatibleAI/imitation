import numpy as np
from gym import utils

from imitation.envs.examples.airl_envs.common import get_asset_xml
from imitation.envs.examples.airl_envs.twod_mjc_env import TwoDEnv

INIT_POS = np.array([0.15, 0.15])
TARGET = np.array([0.15, -0.15])
DIST_THRESH = 0.12


class TwoDMaze(TwoDEnv, utils.EzPickle):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.max_episode_length = 200
        self.episode_length = 0
        utils.EzPickle.__init__(self)
        TwoDEnv.__init__(
            self,
            get_asset_xml("twod_maze.xml"),
            2,
            xbounds=[-0.3, 0.3],
            ybounds=[-0.3, 0.3],
        )

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        pos = ob[0:2]
        dist = np.sum(np.abs(pos - TARGET))  # np.linalg.norm(pos - TARGET)
        reward = -(dist)

        reward_ctrl = -np.square(a).sum()
        reward += 1e-3 * reward_ctrl

        if self.verbose:
            print(pos, reward)
        self.episode_length += 1
        done = self.episode_length >= self.max_episode_length
        return ob, reward, done, {"distance": dist}

    def reset_model(self):
        self.episode_length = 0
        qpos = self.init_qpos + self.np_random.uniform(
            size=self.model.nq, low=-0.01, high=0.01
        )
        qvel = self.init_qvel + self.np_random.uniform(
            size=self.model.nv, low=-0.01, high=0.01
        )
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        # return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()
        return np.concatenate([self.sim.data.qpos]).ravel() - INIT_POS

    def viewer_setup(self):
        pass
