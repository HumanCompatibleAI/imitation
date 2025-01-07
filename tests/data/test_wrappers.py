"""Tests for `imitation.data.wrappers`."""

from typing import Dict, List, Optional, Sequence

import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.data import types
from imitation.data.wrappers import HR_OBS_KEY, BufferingWrapper, HumanReadableWrapper


class _CountingEnv(gym.Env):  # pragma: no cover
    """At timestep `t` of each episode, has `reward / 10 == obs == t`.

    Episodes finish after `episode_length` calls to `step()`. As an example,
    if we have `episode_length=5`, then an episode is to have the
    following observations and rewards:

    ```
    obs = [0, 1, 2, 3, 4, 5]
    rews = [10, 20, 30, 40, 50]
    ```
    """

    def __init__(self, episode_length: int = 5, render_mode: Optional[str] = None):
        assert episode_length >= 1
        self.observation_space = gym.spaces.Box(low=0, high=np.inf, shape=())
        self.action_space = gym.spaces.Box(low=0, high=np.inf, shape=())
        self.episode_length = episode_length
        self.timestep: Optional[int] = None
        self._render_mode = render_mode

    @property
    def render_mode(self):
        return self._render_mode

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        t, self.timestep = 0, 1
        return t, {}

    def step(self, action):
        if self.timestep is None:
            raise RuntimeError("Need to reset before first step().")
        if self.timestep > self.episode_length:
            raise RuntimeError("Episode is over. Need to step().")
        if np.array(action) not in self.action_space:
            raise ValueError(f"Invalid action {action}")

        t, self.timestep = self.timestep, self.timestep + 1
        done = t == self.episode_length
        return t, t * 10, done, False, {}

    def render(self):
        if self._render_mode != "rgb_array":
            raise ValueError(f"Invalid render mode {self._render_mode}")
        return np.array([self.timestep] * 10)


class _CountingDictEnv(_CountingEnv):  # pragma: no cover
    """Similar to _CountingEnv, but with Dict observation."""

    def __init__(self, episode_length: int = 5, render_mode: Optional[str] = None):
        super().__init__(episode_length, render_mode)
        self.observation_space = gym.spaces.Dict(
            spaces={
                "t": gym.spaces.Box(low=0, high=np.inf, shape=()),
                "2t": gym.spaces.Box(low=0, high=np.inf, shape=()),
            },
        )

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        t, self.timestep = 0, 1
        return {"t": t, "2t": 2 * t}, {}

    def step(self, action):
        if self.timestep is None:
            raise RuntimeError("Need to reset before first step().")
        if self.timestep > self.episode_length:
            raise RuntimeError("Episode is over. Need to step().")
        if np.array(action) not in self.action_space:
            raise ValueError(f"Invalid action {action}")

        t, self.timestep = self.timestep, self.timestep + 1
        done = t == self.episode_length
        return {"t": t, "2t": 2 * t}, t * 10, done, False, {}


Envs = [_CountingEnv, _CountingDictEnv]


def _make_buffering_venv(
    Env,
    error_on_premature_reset: bool,
) -> BufferingWrapper:
    venv = DummyVecEnv([Env] * 2)
    venv = DummyVecEnv([Env] * 2)
    wrapped_venv = BufferingWrapper(venv, error_on_premature_reset)
    wrapped_venv.reset()
    return wrapped_venv


def _assert_equal_scrambled_vectors(a: np.ndarray, b: np.ndarray) -> None:
    """Raises AssertionError if `a` and `b` are not identical up to sorting."""
    assert a.shape == b.shape
    assert a.ndim == 1
    np.testing.assert_allclose(np.sort(a), np.sort(b))


def _join_transitions(
    trans_list: Sequence[types.TransitionsWithRew],
) -> types.TransitionsWithRew:
    def concat(x):
        return np.concatenate(list(x))

    obs = concat(t.obs for t in trans_list)
    next_obs = concat(t.next_obs for t in trans_list)
    rews = concat(t.rews for t in trans_list)
    acts = concat(t.acts for t in trans_list)
    dones = concat(t.dones for t in trans_list)
    infos = concat(t.infos for t in trans_list)
    return types.TransitionsWithRew(
        obs=obs,
        next_obs=next_obs,
        rews=rews,
        acts=acts,
        dones=dones,
        infos=infos,
    )


@pytest.mark.parametrize("Env", Envs)
@pytest.mark.parametrize("episode_lengths", [(1,), (6, 5, 1, 2), (2, 2)])
@pytest.mark.parametrize("n_steps", [1, 2, 20, 21])
@pytest.mark.parametrize("extra_pop_timesteps", [(), (1,), (4, 8)])
def test_pop(
    Env,
    episode_lengths: Sequence[int],
    n_steps: int,
    extra_pop_timesteps: Sequence[int],
) -> None:
    """Check pop_transitions() results for BufferWrapper.

    To make things easier to test, we use _CountingEnv where the observation
    is simply the episode timestep. The reward is 10x the timestep. Our action
    is 2.1x the timestep. There is an confusing offset for the observation because
    it has timestep 0 (due to reset()) and the other quantities don't, so here is
    an example of environment outputs and associated actions:

    ```
    episode_length = 5
    obs = [0, 1, 2, 3, 4, 5]  (len=6)
    acts = [0, 2.1, 4.2, ..., 8.4]  (len=5)
    rews = [10, ..., 50]  (len=5)
    ```

    Converted to `Transition`-format, this looks like:
    ```
    episode_length = 5
    obs = [0, 1, 2, 3, 4, 5]  (len=5)
    next_obs = [1, 2, 3, 4, 5]  (len=5)
    acts = [0, 2.1, 4.2, ..., 8.4]  (len=5)
    rews = [10, ..., 50]  (len=5)
    ```

    Args:
        Env: Environment class type.
        Env: Environment class type.
        episode_lengths: The number of timesteps before episode end in each dummy
            environment.
        n_steps: Number of times to call `step()` on the dummy environment.
        extra_pop_timesteps: By default, we only call `pop_*()` after `n_steps` calls
            to `step()`. For every unique positive `x` in `extra_pop_timesteps`, we
            also call `pop_*()` after the `x`th call to `step()`. All popped samples
            are concatenated before validating results at the end of this test case.
            All `x` in `extra_pop_timesteps` must be in range(1, n_steps). (`x == 0`
            is not valid because there are no transitions to pop at timestep 0).

    Raises:
        ValueError: `n_steps <= 0`.
    """
    if not n_steps >= 1:  # pragma: no cover
        raise ValueError(f"n_steps = {n_steps} <= 0")
    for t in extra_pop_timesteps:  # pragma: no cover
        if t < 1:
            raise ValueError(t)
        if not 1 <= t < n_steps:
            pytest.skip("pop timesteps out of bounds for this test case")

    def make_env(ep_len):
        return lambda: Env(episode_length=ep_len)
        return lambda: Env(episode_length=ep_len)

    venv = DummyVecEnv([make_env(ep_len) for ep_len in episode_lengths])
    venv_buffer = BufferingWrapper(venv)

    # To test `pop_transitions`, we will check that every obs, act, and rew
    # returned by `.reset()` and `.step()` is also returned by one of the
    # calls to `pop_transitions()`.
    transitions_list: List[types.TransitionsWithRew] = []

    # Initial observation (only matters for pop_transitions()).
    obs = venv_buffer.reset()
    if Env == _CountingEnv:
        np.testing.assert_array_equal(obs, [0] * venv.num_envs)
    else:
        np.testing.assert_array_equal(obs["t"], [0] * venv.num_envs)
    if Env == _CountingEnv:
        np.testing.assert_array_equal(obs, [0] * venv.num_envs)
    else:
        np.testing.assert_array_equal(obs["t"], [0] * venv.num_envs)

    for t in range(1, n_steps + 1):
        acts = obs * 2.1 if Env == _CountingEnv else obs["t"] * 2.1
        acts = obs * 2.1 if Env == _CountingEnv else obs["t"] * 2.1
        venv_buffer.step_async(acts)
        obs, *_ = venv_buffer.step_wait()

        if t in extra_pop_timesteps:
            transitions_list.append(venv_buffer.pop_transitions())

    transitions_list.append(venv_buffer.pop_transitions())

    # Build expected transitions
    expect_obs_list = []
    for ep_len in episode_lengths:
        n_complete, remainder = divmod(n_steps, ep_len)
        expect_obs_list.extend([np.arange(ep_len)] * n_complete)
        expect_obs_list.append(np.arange(remainder))

    expect_obs = np.concatenate(expect_obs_list)
    expect_next_obs = expect_obs + 1
    expect_acts = expect_obs * 2.1
    expect_rews = expect_next_obs * 10

    # Check `pop_transitions()`
    trans = _join_transitions(transitions_list)
    if Env == _CountingEnv:
        actual_obs = types.assert_not_dictobs(trans.obs)
        actual_next_obs = types.assert_not_dictobs(trans.next_obs)
    else:
        actual_obs = types.DictObs.stack(trans.obs).get("t")
        actual_next_obs = types.DictObs.stack(trans.next_obs).get("t")
    _assert_equal_scrambled_vectors(actual_obs, expect_obs)
    _assert_equal_scrambled_vectors(actual_next_obs, expect_next_obs)
    if Env == _CountingEnv:
        actual_obs = types.assert_not_dictobs(trans.obs)
        actual_next_obs = types.assert_not_dictobs(trans.next_obs)
    else:
        actual_obs = types.DictObs.stack(trans.obs).get("t")
        actual_next_obs = types.DictObs.stack(trans.next_obs).get("t")
    _assert_equal_scrambled_vectors(actual_obs, expect_obs)
    _assert_equal_scrambled_vectors(actual_next_obs, expect_next_obs)
    _assert_equal_scrambled_vectors(trans.acts, expect_acts)
    _assert_equal_scrambled_vectors(trans.rews, expect_rews)


@pytest.mark.parametrize("Env", Envs)
def test_reset_error(Env):
    # Resetting before a `step()` is okay.
    for flag in [True, False]:
        venv = _make_buffering_venv(Env, flag)
        venv = _make_buffering_venv(Env, flag)
        for _ in range(10):
            venv.reset()

    # Resetting after a `step()` is not okay if error flag is True.
    venv = _make_buffering_venv(Env, True)
    venv = _make_buffering_venv(Env, True)
    zeros = np.array([0.0, 0.0], dtype=venv.action_space.dtype)
    venv.step(zeros)
    with pytest.raises(RuntimeError, match="before samples were accessed"):
        venv.reset()

    # Same as previous case, but insert a `pop_transitions()` in between.
    venv = _make_buffering_venv(Env, True)
    venv = _make_buffering_venv(Env, True)
    venv.step(zeros)
    venv.pop_transitions()
    venv.step(zeros)
    with pytest.raises(RuntimeError, match="before samples were accessed"):
        venv.reset()

    # Resetting after a `step()` is ok if error flag is False.
    venv = _make_buffering_venv(Env, False)
    venv.step(zeros)
    venv.reset()

    # Resetting after a `step()` is ok if transitions are first collected.
    for flag in [True, False]:
        venv = _make_buffering_venv(Env, flag)
        venv = _make_buffering_venv(Env, flag)
        venv.step(zeros)
        venv.pop_transitions()
        venv.reset()


@pytest.mark.parametrize("Env", Envs)
def test_n_transitions_and_empty_error(Env):
    venv = _make_buffering_venv(Env, True)
    trajs, ep_lens = venv.pop_trajectories()
    assert trajs == []
    assert ep_lens == []
    zeros = np.array([0.0, 0.0], dtype=venv.action_space.dtype)
    venv.step(zeros)
    assert venv.n_transitions == 2
    venv.step(zeros)
    assert venv.n_transitions == 4
    venv.pop_transitions()
    assert venv.n_transitions == 0
    with pytest.raises(RuntimeError, match=".* empty .*"):
        venv.pop_transitions()


@pytest.mark.parametrize("Env", Envs)
@pytest.mark.parametrize("original_obs_key", ["k1", "k2"])
def test_human_readable_wrapper(Env, original_obs_key):
    num_obs_key_expected = 2 if Env is _CountingEnv else 3
    origin_obs_key = original_obs_key if Env is _CountingEnv else "t"
    ori_env = Env(render_mode="rgb_array")
    hr_env = HumanReadableWrapper(
        ori_env,
        original_obs_key=original_obs_key,
    )
    assert hr_env.observation_space == ori_env.observation_space

    obs, _ = hr_env.reset()
    assert isinstance(obs, Dict)
    assert HR_OBS_KEY in obs
    assert len(obs) == num_obs_key_expected
    assert obs[origin_obs_key] == 0
    _assert_equal_scrambled_vectors(obs[HR_OBS_KEY], np.array([1] * 10))

    next_obs, *_ = hr_env.step(hr_env.action_space.sample())
    assert isinstance(next_obs, Dict)
    assert HR_OBS_KEY in next_obs
    assert len(next_obs) == num_obs_key_expected
    assert next_obs[origin_obs_key] == 1
    _assert_equal_scrambled_vectors(next_obs[HR_OBS_KEY], np.array([2] * 10))
