import numpy as np
from gym import utils
from gym.envs.mujoco import mujoco_env

from imitation.envs.examples.airl_envs.dynamic_mjc.mjc_models import pusher


class PusherEnv(mujoco_env.MujocoEnv, utils.EzPickle):
    def __init__(self, sparse_reward=False, no_reward=False, episode_length=200):
        utils.EzPickle.__init__(self)
        self.sparse_reward = sparse_reward
        self.no_reward = no_reward
        self.max_episode_length = episode_length
        self.goal_pos = np.asarray([0.0, 0.0])

        self.episode_length = 0

        model = pusher(goal_pos=[self.goal_pos[0], self.goal_pos[1], -0.323])
        with model.asfile() as f:
            mujoco_env.MujocoEnv.__init__(self, f.name, 5)

    def step(self, a):
        vec_1 = self.get_body_com("object") - self.get_body_com("tips_arm")
        vec_2 = self.get_body_com("object") - self.get_body_com("goal")

        reward_near = -np.linalg.norm(vec_1)  # arm to object
        reward_dist = -np.linalg.norm(vec_2[0:2])  # object to goal
        reward_ctrl = -np.square(a).sum()
        if self.no_reward:
            reward = 0
        elif self.sparse_reward:
            reward = reward_dist + 0.1 * reward_ctrl
        else:
            reward = reward_dist + 0.1 * reward_ctrl + 0.5 * reward_near

        self.do_simulation(a, self.frame_skip)
        ob = self._get_obs()
        self.episode_length += 1
        done = self.episode_length >= self.max_episode_length
        return (
            ob,
            reward,
            done,
            dict(
                reward_dist=reward_dist,
                reward_near=reward_near,
                reward_ctrl=reward_ctrl,
            ),
        )

    def viewer_setup(self):
        self.viewer.cam.trackbodyid = -1
        self.viewer.cam.distance = 4.0

    def reset_model(self):
        qpos = self.init_qpos
        self.episode_length = 0

        while True:
            self.cylinder_pos = np.concatenate(
                [
                    self.np_random.uniform(low=-0.5, high=0, size=1),
                    self.np_random.uniform(low=-0.5, high=0.5, size=1),
                ]
            )
            cyl_dist = np.linalg.norm(self.cylinder_pos - self.goal_pos)
            if cyl_dist > 0.2 and cyl_dist < 0.4:
                break

        qpos[-4:-2] = self.cylinder_pos
        qpos[-2] = self.goal_pos[0]
        qpos[-1] = self.goal_pos[1]
        qvel = self.init_qvel + self.np_random.uniform(
            low=-0.005, high=0.005, size=self.model.nv
        )
        qvel[-4:] = 0
        self.set_state(qpos, qvel)
        return self._get_obs()

    def _get_obs(self):
        return np.concatenate(
            [
                self.sim.data.qpos.flat[:7],
                self.sim.data.qvel.flat[:7],
                self.get_body_com("tips_arm"),
                self.get_body_com("object"),
                self.get_body_com("goal"),
            ]
        )

    def plot_trajs(self, *args, **kwargs):
        pass
