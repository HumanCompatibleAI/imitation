import itertools
import random
from typing import List

import gym
import numpy as np
from gym.spaces import Discrete
from mazelab import BaseEnv, BaseMaze
from mazelab import DeepMindColor as color
from mazelab import Object, VonNeumannNoOpMotion


class Maze(BaseMaze):
    def __init__(self, maze_array: np.ndarray):
        self.data = maze_array
        super().__init__()

    @property
    def size(self):
        return self.data.shape

    def make_objects(self):
        free = Object(
            "free",
            0,
            color.free,
            False,
            np.stack(np.where(self.data == 0), axis=1),
        )
        obstacle = Object(
            "obstacle",
            1,
            color.obstacle,
            True,
            np.stack(np.where(self.data == 1), axis=1),
        )
        agent = Object("agent", 2, color.agent, False, [])
        goal = Object("goal", 3, color.goal, False, [])
        return free, obstacle, agent, goal


class MazeEnv(BaseEnv):
    def __init__(
        self,
        size: int = 10,
        random_start: bool = True,
        reward: str = "goal",
        shaping: str = "zero",
        gamma: float = 0.99,
    ):
        # among other things, this calls self.seed() so that the self.rng
        # object exists
        super().__init__()
        x = np.zeros((size, size))
        self.size = size
        self.start_idx = [[1, 1]]
        self.goal_idx = [[8, 8]]
        self.random_start = random_start
        self.gamma = gamma
        self.rewards = np.zeros((size ** 2, size ** 2))
        goal_pos = size * self.goal_idx[0][0] + self.goal_idx[0][1]
        if reward == "goal":
            self.rewards[goal_pos, :] = 1.0
        elif reward == "path":
            self.rewards[:, :] = -0.2
            diagonal = size * np.arange(size) + np.arange(size)
            self.rewards[diagonal, :] = 0.0
            self.rewards[goal_pos, :] = 1.0
        else:
            raise ValueError(f"Unknown reward type {reward}")

        if shaping == "zero":
            pass
        elif shaping == "dense":
            for i, j in itertools.product(range(size), repeat=2):
                pos = self._to_idx((i, j))

                current_potential = -(
                    abs(i - self.goal_idx[0][0]) + abs(j - self.goal_idx[0][1])
                )

                next_potential = -(
                    abs(i - self.goal_idx[0][0]) + abs(j - self.goal_idx[0][1])
                )

                self.rewards[pos, :] -= current_potential
                self.rewards[:, pos] += self.gamma * next_potential
        elif shaping == "antidense":
            for i, j in itertools.product(range(size), repeat=2):
                pos = self._to_idx((i, j))

                current_potential = abs(i - self.goal_idx[0][0]) + abs(
                    j - self.goal_idx[0][1]
                )

                next_potential = abs(i - self.goal_idx[0][0]) + abs(
                    j - self.goal_idx[0][1]
                )

                self.rewards[pos, :] -= current_potential
                self.rewards[:, pos] += self.gamma * next_potential
        else:
            raise ValueError(f"Unknown shaping type {shaping}")

        self.maze = Maze(x)
        self.motions = VonNeumannNoOpMotion()

        self.observation_space = Discrete(size ** 2)
        self.action_space = Discrete(len(self.motions))

    def seed(self, seed: int = 0) -> List[int]:
        super().seed(seed)
        self.rng = random.Random(seed)
        return [seed]

    def step(self, action):
        motion = self.motions[action]
        current_position = self.maze.objects.agent.positions[0]
        new_position = [
            current_position[0] + motion[0],
            current_position[1] + motion[1],
        ]
        valid = self._is_valid(new_position)
        if valid:
            self.maze.objects.agent.positions = [new_position]
        else:
            new_position = current_position

        done = False
        reward = self._reward(current_position, action, new_position)
        return self._get_obs(), reward, done, {}

    def _reward(self, state, action, next_state) -> float:
        return self.rewards[self._to_idx(state), self._to_idx(next_state)]

    def _to_idx(self, position):
        return self.size * position[0] + position[1]

    def _get_obs(self):
        current_position = self.maze.objects.agent.positions[0]
        return self._to_idx(current_position)

    def reset(self):
        self.maze.objects.goal.positions = self.goal_idx
        if self.random_start:
            available_positions = [
                pos
                # free positions are stored as a numpy array, we need a list
                # to compare to goal position
                for pos in self.maze.objects.free.positions.tolist()
                # The "free" object positions are not all empty tiles!
                # Multiple objects can be at one position, and "free"
                # just means that there is no wall there, but the goal
                # might still be on this field. So we need to filter that
                # out because the agent shouldn't start on top of the goal.
                if pos not in self.maze.objects.goal.positions
            ]
            self.maze.objects.agent.positions = [
                list(self.rng.choice(available_positions)),
            ]
        else:
            self.maze.objects.agent.positions = [self.start_idx]
        return self._get_obs()

    def _is_valid(self, position):
        # position indices must be non-negative
        if position[0] < 0 or position[1] < 0:
            return False
        # position indices must not be out of bounds
        if position[0] >= self.maze.size[0] or position[1] >= self.maze.size[1]:
            return False
        # position must be passable
        if self.maze.to_impassable()[position[0]][position[1]]:
            return False
        return True

    def _is_goal(self, position):
        out = False
        for pos in self.maze.objects.goal.positions:
            if position[0] == pos[0] and position[1] == pos[1]:
                out = True
                break
        return out

    def get_image(self):
        return self.maze.to_rgb()


gym.register(
    "imitation/EmptyMaze-v0",
    entry_point=MazeEnv,
    max_episode_steps=20,
)

gym.register(
    "imitation/EmptyMazeDense-v0",
    entry_point=MazeEnv,
    max_episode_steps=20,
    kwargs={"shaping": "dense"},
)

gym.register(
    "imitation/EmptyMazeAntiDense-v0",
    entry_point=MazeEnv,
    max_episode_steps=20,
    kwargs={"shaping": "antidense"},
)

gym.register(
    "imitation/EmptyMazePath-v0",
    entry_point=MazeEnv,
    max_episode_steps=20,
    kwargs={"reward": "path"},
)

gym.register(
    "imitation/EmptyMazePathDense-v0",
    entry_point=MazeEnv,
    max_episode_steps=20,
    kwargs={"reward": "path", "shaping": "dense"},
)
