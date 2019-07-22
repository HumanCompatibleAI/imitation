import logging

from gym.envs import register

register(id='airl/ObjPusher-v0', entry_point='airl.envs.pusher_env:PusherEnv', kwargs={'sparse_reward': False})
register(id='airl/TwoDMaze-v0', entry_point='airl.envs.twod_maze:TwoDMaze')
register(id='airl/PointMazeRight-v0', entry_point='airl.envs.point_maze_env:PointMazeEnv',
         kwargs={'sparse_reward': False, 'direction': 1})
register(id='airl/PointMazeLeft-v0', entry_point='airl.envs.point_maze_env:PointMazeEnv',
         kwargs={'sparse_reward': False, 'direction': 0})

# A modified ant which flips over less and learns faster via TRPO
register(id='airl/CustomAnt-v0', entry_point='airl.envs.ant_env:CustomAntEnv',
         kwargs={'gear': 30, 'disabled': False})
register(id='airl/DisabledAnt-v0', entry_point='airl.envs.ant_env:CustomAntEnv',
         kwargs={'gear': 30, 'disabled': True})

register(id='airl/VisualPointMaze-v0', entry_point='airl.envs.visual_pointmass:VisualPointMazeEnv',
         kwargs={'sparse_reward': False, 'direction': 1})
