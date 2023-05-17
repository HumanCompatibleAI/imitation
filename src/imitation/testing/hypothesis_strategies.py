"""Hypothesis strategies for generating sequences of trajectories for testing."""
from typing import Dict, List, Union

import gym
import numpy as np
from hypothesis import strategies as st

from imitation.data import types

gym_spaces = st.sampled_from(
    [
        gym.spaces.Discrete(3),
        gym.spaces.MultiDiscrete((3, 4)),
        gym.spaces.Box(-1, 1, shape=(1,)),
        gym.spaces.Box(-1, 1, shape=(2,)),
        gym.spaces.Box(-np.inf, np.inf, shape=(2,)),
    ],
)
"""A strategy to generate spaces supported by trajectory serialization."""

info_dict_contents = st.dictionaries(
    keys=st.text(),
    values=st.one_of(
        st.integers(),
        st.floats(allow_nan=False),
        st.text(),
        st.lists(st.floats(allow_nan=False)),
    ),
)
"""A strategy to generate contents of the info dict for a trajectory."""

trajectory_length = st.integers(min_value=1, max_value=10)
"""The length of a trajectories we want to test."""


def _build_trajectory_without_reward(
    obs_space: gym.Space,
    act_space: gym.Space,
    length: int,
    info_dict_contents: Dict[str, Union[int, str, float, List[float]]],
    terminal: bool,
):
    return types.Trajectory(
        obs=np.array([obs_space.sample() for _ in range(length + 1)]),
        acts=np.array([act_space.sample() for _ in range(length)]),
        infos=np.array([info_dict_contents for _ in range(length)]),
        terminal=terminal,
    )


def _build_trajectory_with_rew(
    obs_space: gym.Space,
    act_space: gym.Space,
    length: int,
    info_dict_contents: Dict[str, Union[int, str, float, List[float]]],
    terminal: bool,
    min_rew: float,
    max_rew: float,
):
    return types.TrajectoryWithRew(
        obs=np.array([obs_space.sample() for _ in range(length + 1)]),
        acts=np.array([act_space.sample() for _ in range(length)]),
        infos=np.array([info_dict_contents for _ in range(length)]),
        terminal=terminal,
        rews=np.random.uniform(min_rew, max_rew, size=length),
    )


# Note: those shared strategies are used to ensure that each trajectory in a list
# is generated using the same spaces.
_shared_obs_space = st.shared(gym_spaces, key="obs_space")
_shared_act_space = st.shared(gym_spaces, key="act_space")

trajectories_without_reward_list = st.lists(
    st.builds(
        _build_trajectory_without_reward,
        obs_space=_shared_obs_space,
        act_space=_shared_act_space,
        length=trajectory_length,
        info_dict_contents=info_dict_contents,
        terminal=st.booleans(),
    ),
    min_size=1,
    max_size=10,
)
"""A strategy to generate lists of trajectories (without reward) for testing.

All trajectories in the list are generated using the same spaces.
"""

trajectories_with_reward_list = st.lists(
    st.builds(
        _build_trajectory_with_rew,
        obs_space=_shared_obs_space,
        act_space=_shared_act_space,
        length=trajectory_length,
        info_dict_contents=info_dict_contents,
        terminal=st.booleans(),
        min_rew=st.floats(min_value=-100, max_value=100),
        max_rew=st.floats(min_value=-100, max_value=100),
    ),
    min_size=1,
    max_size=10,
)
"""A strategy to generate lists of trajectories (with reward) for testing.

All trajectories in the list are generated using the same spaces.
"""

trajectories_list = st.one_of(
    trajectories_without_reward_list,
    trajectories_with_reward_list,
)
"""A strategy to generate lists of trajectories (with or without reward) for testing.

All trajectories in the list are generated using the same spaces.
They either all have reward or none of them have reward.
"""
