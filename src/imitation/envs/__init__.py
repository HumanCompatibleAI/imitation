from typing import Optional
from gym.envs import register as gym_register

ENV_NAMES = []

def _register(id: str, entry_point: str, kwargs: Optional[dict] = None):
    gym_register(id=id, entry_point=entry_point, kwargs=kwargs)
    ENV_NAMES.append(id)


_register(id='imitation/ObjPusher-v0',
          entry_point='imitation.envs.pusher_env:PusherEnv',
          kwargs={'sparse_reward': False})
_register(id='imitation/TwoDMaze-v0',
          entry_point='imitation.envs.twod_maze:TwoDMaze')
_register(id='imitation/PointMazeRight-v0',
          entry_point='imitation.envs.point_maze_env:PointMazeEnv',
          kwargs={
              'sparse_reward': False,
              'direction': 1,
          })
_register(id='imitation/PointMazeLeft-v0',
          entry_point='imitation.envs.point_maze_env:PointMazeEnv',
          kwargs={
              'sparse_reward': False,
              'direction': 0,
          })

# A modified ant which flips over less and learns faster via TRPO
_register(id='imitation/CustomAnt-v0',
          entry_point='imitation.envs.ant_env:CustomAntEnv',
          kwargs={
              'gear': 30,
              'disabled': False,
          })
_register(id='imitation/DisabledAnt-v0',
          entry_point='imitation.envs.ant_env:CustomAntEnv',
          kwargs={
              'gear': 30,
              'disabled': True,
          })

# FIXME: There are some import issues with VisualPointMaze, not sure if
# we even want to keep this right now.
# (Need to install `cv2`. Need to to edit VisualPointMaze's _get_viewer() to
# use with mode='rgb_array' or mode='depth_array'.  Not sure which.)
# https://github.com/openai/gym/blob/master/gym/envs/mujoco/mujoco_env.py
#
# _register(id='imitation/VisualPointMaze-v0',
#           entry_point='imitation.envs.visual_pointmass:VisualPointMazeEnv',
#           kwargs={
#               'sparse_reward': False,
#               'direction': 1,
#           })
