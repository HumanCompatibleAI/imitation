"""Tests for imitation.data.huggingface_utils."""
from typing import Dict, List, Union

import gym
import hypothesis
import hypothesis.strategies as st
import numpy as np

from imitation.data import serialize, types

# Make hypothesis strategy for spaces
spaces = st.sampled_from(
    [
        gym.spaces.Discrete(3),
        gym.spaces.MultiDiscrete((3, 4)),
        gym.spaces.Box(-1, 1, shape=(1,)),
        gym.spaces.Box(-1, 1, shape=(2,)),
        gym.spaces.Box(-np.inf, np.inf, shape=(2,)),
    ],
)

info_dict_contents = st.dictionaries(
    keys=st.text(),
    values=st.one_of(
        st.integers(),
        st.floats(allow_nan=False),
        st.text(),
        st.lists(st.floats(allow_nan=False)),
    ),
)

trajectory_length = st.integers(min_value=1, max_value=10)


def build_trajectory(
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


trajectories = st.builds(
    build_trajectory,
    obs_space=spaces,
    act_space=spaces,
    length=st.integers(min_value=1, max_value=10),
    info_dict_contents=info_dict_contents,
    terminal=st.booleans(),
)


@st.composite
def trajectories_list(
    draw,
    obs_space=spaces,
    act_space=spaces,
    length=st.integers(min_value=1, max_value=10),
):
    the_obs_space = draw(obs_space)
    the_act_space = draw(act_space)
    return [
        build_trajectory(
            obs_space=the_obs_space,
            act_space=the_act_space,
            length=draw(trajectory_length),
            info_dict_contents=draw(info_dict_contents),
            terminal=draw(st.booleans()),
        )
        for _ in range(draw(length))
    ]


def build_trajectory_with_rew(
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


trajectories_with_rew = st.builds(
    build_trajectory_with_rew,
    obs_space=spaces,
    act_space=spaces,
    length=trajectory_length,
    info_dict_contents=info_dict_contents,
    terminal=st.booleans(),
    min_rew=st.floats(min_value=-100, max_value=100),
    max_rew=st.floats(min_value=-100, max_value=100),
)


@st.composite
def trajectories_with_rew_list(
    draw,
    obs_space=spaces,
    act_space=spaces,
    length=st.integers(min_value=1, max_value=10),
):
    the_obs_space = draw(obs_space)
    the_act_space = draw(act_space)
    return [
        build_trajectory_with_rew(
            obs_space=the_obs_space,
            act_space=the_act_space,
            length=draw(trajectory_length),
            info_dict_contents=draw(info_dict_contents),
            terminal=draw(st.booleans()),
            min_rew=draw(st.floats(min_value=-100, max_value=100)),
            max_rew=draw(st.floats(min_value=-100, max_value=100)),
        )
        for _ in range(draw(length))
    ]


trajectories_list_with_or_without_rew = st.one_of(
    trajectories_list(),
    trajectories_with_rew_list(),
)


@hypothesis.given(trajectories=trajectories_list_with_or_without_rew)
def test_save_load_roundtrip(trajectories):
    serialize.save("/tmp/haha", trajectories)
    loaded_trajectories = serialize.load("/tmp/haha")

    assert len(trajectories) == len(loaded_trajectories)
    for traj, loaded_traj in zip(trajectories, loaded_trajectories):
        assert traj == loaded_traj
