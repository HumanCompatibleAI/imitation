import logging

from gym.envs import register

register(id='imitation/ObjPusher-v0',
         entry_point='imitation.envs.pusher_env:PusherEnv',
         kwargs={'sparse_reward': False})
register(id='imitation/TwoDMaze-v0',
         entry_point='imitation.envs.twod_maze:TwoDMaze')
register(id='imitation/PointMazeRight-v0',
         entry_point='imitation.envs.point_maze_env:PointMazeEnv',
         kwargs={'sparse_reward': False, 'direction': 1})
register(id='imitation/PointMazeLeft-v0',
         entry_point='imitation.envs.point_maze_env:PointMazeEnv',
         kwargs={'sparse_reward': False, 'direction': 0})

# A modified ant which flips over less and learns faster via TRPO
register(id='imitation/CustomAnt-v0',
         entry_point='imitation.envs.ant_env:CustomAntEnv',
         kwargs={'gear': 30, 'disabled': False})
register(id='imitation/DisabledAnt-v0',
         entry_point='imitation.envs.ant_env:CustomAntEnv',
         kwargs={'gear': 30, 'disabled': True})

register(id='imitation/VisualPointMaze-v0',
         entry_point='imitation.envs.visual_pointmass:VisualPointMazeEnv',
         kwargs={'sparse_reward': False, 'direction': 1})
