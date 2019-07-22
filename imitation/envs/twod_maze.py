import numpy as np
from gym import utils

from airl.envs.env_utils import get_asset_xml
from airl.envs.twod_mjc_env import TwoDEnv

from rllab.misc import logger as logger

INIT_POS = np.array([0.15,0.15])
TARGET = np.array([0.15, -0.15])
DIST_THRESH = 0.12

class TwoDMaze(TwoDEnv, utils.EzPickle):
    def __init__(self, verbose=False):
        self.verbose = verbose
        self.max_episode_length = 200
        self.episode_length = 0
        utils.EzPickle.__init__(self)
        TwoDEnv.__init__(self, get_asset_xml('twod_maze.xml'), 2, xbounds=[-0.3,0.3], ybounds=[-0.3,0.3])

    def step(self, a):
        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        pos = ob[0:2]
        dist = np.sum(np.abs(pos-TARGET)) #np.linalg.norm(pos - TARGET)
        reward = - (dist)

        reward_ctrl = - np.square(a).sum()
        reward += 1e-3 * reward_ctrl

        if self.verbose:
            print(pos, reward)
        self.episode_length += 1
        done = self.episode_length >= self.max_episode_length
        return ob, reward, done, {'distance': dist}

    def reset_model(self):
        self.episode_length = 0
        qpos = self.init_qpos + self.np_random.uniform(size=self.model.nq, low=-0.01, high=0.01)
        qvel = self.init_qvel + self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        #return np.concatenate([self.sim.data.qpos, self.sim.data.qvel]).ravel()
        return np.concatenate([self.sim.data.qpos]).ravel() - INIT_POS

    def viewer_setup(self):
        v = self.viewer
        #v.cam.trackbodyid=0
        #v.cam.distance = v.model.stat.extent

    def log_diagnostics(self, paths):
        rew_dist = np.array([traj['env_infos']['distance'] for traj in paths])

        logger.record_tabular('AvgObjectToGoalDist', np.mean(rew_dist))
        logger.record_tabular('MinAvgObjectToGoalDist', np.mean(np.min(rew_dist, axis=1)))



if __name__ == "__main__":
    from airl.utils.getch import getKey
    env = TwoDMaze(verbose=True)

    while True:
        key = getKey()
        a = np.array([0.0,0.0])
        if key == 'w':
            a += np.array([0.0, 1.0])
        elif key == 'a':
            a += np.array([-1.0, 0.0])
        elif key  == 's':
            a += np.array([0.0, -1.0])
        elif key  == 'd':
            a += np.array([1.0, 0.0])
        elif key  == 'q':
            break
        a *= 0.2
        env.step(a)
        env.render()