from typing import Optional

from gym.envs import register as gym_register

ENV_NAMES = []
_ENTRY_POINT_PREFIX = 'imitation.examples.airl_envs'


def _register(env_name: str, entry_point: str, kwargs: Optional[dict] = None):
    entry_point = f"{_ENTRY_POINT_PREFIX}.{entry_point}"
    gym_register(id=env_name, entry_point=entry_point, kwargs=kwargs)
    ENV_NAMES.append(env_name)


_register('imitation/ObjPusher-v0',
          entry_point=f'pusher_env:PusherEnv',
          kwargs={'sparse_reward': False})
_register('imitation/TwoDMaze-v0',
          entry_point='twod_maze:TwoDMaze')
_register('imitation/PointMazeRight-v0',
          entry_point='point_maze_env:PointMazeEnv',
          kwargs={
              'sparse_reward': False,
              'direction': 1,
          })
_register('imitation/PointMazeLeft-v0',
          entry_point='point_maze_env:PointMazeEnv',
          kwargs={
              'sparse_reward': False,
              'direction': 0,
          })

# A modified ant which flips over less and learns faster via TRPO
_register('imitation/CustomAnt-v0',
          entry_point='ant_env:CustomAntEnv',
          kwargs={
              'gear': 30,
              'disabled': False,
          })
_register('imitation/DisabledAnt-v0',
          entry_point='ant_env:CustomAntEnv',
          kwargs={
              'gear': 30,
              'disabled': True,
          })
