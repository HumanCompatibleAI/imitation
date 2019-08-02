from gym import utils
from gym.envs.mujoco import mujoco_env
import numpy as np

from imitation.examples.airl_envs.dynamic_mjc.mjc_models import point_mass_maze


class PointMazeEnv(mujoco_env.MujocoEnv, utils.EzPickle):
  def __init__(self,
               direction=1,
               maze_length=0.6,
               sparse_reward=False,
               no_reward=False,
               episode_length=100):
    utils.EzPickle.__init__(self)
    self.sparse_reward = sparse_reward
    self.no_reward = no_reward
    self.max_episode_length = episode_length
    self.direction = direction
    self.length = maze_length

    self.episode_length = 0

    model = point_mass_maze(direction=self.direction, length=self.length)
    with model.asfile() as f:
      mujoco_env.MujocoEnv.__init__(self, f.name, 5)

  def step(self, a):
    vec_dist = self.get_body_com("particle") - self.get_body_com("target")

    reward_dist = -np.linalg.norm(vec_dist)  # particle to target
    reward_ctrl = -np.square(a).sum()
    if self.no_reward:
      reward = 0
    elif self.sparse_reward:
      if reward_dist <= 0.1:
        reward = 1
      else:
        reward = 0
    else:
      reward = reward_dist + 0.001 * reward_ctrl

    self.do_simulation(a, self.frame_skip)
    ob = self._get_obs()
    self.episode_length += 1
    done = self.episode_length >= self.max_episode_length
    return ob, reward, done, dict(reward_dist=reward_dist,
                                  reward_ctrl=reward_ctrl)

  def viewer_setup(self):
    self.viewer.cam.trackbodyid = -1
    self.viewer.cam.distance = 4.0

  def reset_model(self):
    qpos = self.init_qpos
    self.episode_length = 0
    qvel = self.init_qvel + \
        self.np_random.uniform(size=self.model.nv, low=-0.01, high=0.01)
    self.set_state(qpos, qvel)
    self.episode_length = 0
    return self._get_obs()

  def _get_obs(self):
    return np.concatenate([
        self.get_body_com("particle"),
        # self.get_body_com("target"),
    ])

  def plot_trajs(self, *args, **kwargs):
    pass
