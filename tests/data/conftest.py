import gymnasium as gym
import numpy as np
import pytest

from imitation.data import types

SPACES = [
    gym.spaces.Discrete(3),
    gym.spaces.MultiDiscrete([3, 4]),
    gym.spaces.Box(-1, 1, shape=(1,)),
    gym.spaces.Box(-1, 1, shape=(2,)),
    gym.spaces.Box(-np.inf, np.inf, shape=(2,)),
]
DICT_SPACE = gym.spaces.Dict(
    {"a": gym.spaces.Discrete(3), "b": gym.spaces.Box(-1, 1, shape=(2,))},
)
LENGTHS = [0, 1, 2, 10]


@pytest.fixture(params=SPACES)
def act_space(request):
    return request.param


@pytest.fixture(params=SPACES + [DICT_SPACE])
def obs_space(request):
    return request.param


@pytest.fixture(params=LENGTHS)
def length(request):
    return request.param


@pytest.fixture
def trajectory(
    obs_space: gym.Space,
    act_space: gym.Space,
    length: int,
) -> types.Trajectory:
    """Fixture to generate trajectory of length `length` iid sampled from spaces."""
    if length == 0:
        pytest.skip()

    raw_obs = [obs_space.sample() for _ in range(length + 1)]
    if isinstance(obs_space, gym.spaces.Dict):
        obs: types.Observation = types.DictObs.from_obs_list(raw_obs)
    else:
        obs = np.array(raw_obs)
    acts = np.array([act_space.sample() for _ in range(length)])
    infos = np.array([{f"key{i}": i} for i in range(length)])
    return types.Trajectory(obs=obs, acts=acts, infos=infos, terminal=True)


@pytest.fixture
def trajectory_rew(trajectory: types.Trajectory) -> types.TrajectoryWithRew:
    """Like `trajectory` but with reward randomly sampled from a Gaussian."""
    rews = np.random.randn(len(trajectory))
    return types.TrajectoryWithRew(
        **types.dataclass_quick_asdict(trajectory),
        rews=rews,
    )
