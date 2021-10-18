"""Tests for `imitation.envs.*`."""

import gym
import numpy as np
import pytest

try:
    # seals_test requires mujoco_py, so skip if we don't have that
    from seals.testing import envs as seals_test
except gym.error.DependencyNotInstalled as ex:
    pytest.skip(
        "skipping due to import error on seals.testing, mujoco_py is probably "
        f"missing (error: {ex})",
        allow_module_level=True,
    )
    seals_test = None
from stable_baselines3.common import envs, vec_env

# Unused imports is for side-effect of registering environments
from imitation.envs import examples, resettable_env  # noqa: F401
from imitation.testing import envs as imitation_test

ENV_NAMES = [
    env_spec.id
    for env_spec in gym.envs.registration.registry.all()
    if env_spec.id.startswith("imitation/")
]

DETERMINISTIC_ENVS = []


if seals_test is not None:
    env = pytest.fixture(seals_test.make_env_fixture(skip_fn=pytest.skip))


@pytest.mark.parametrize("env_name", ENV_NAMES)
class TestEnvs:
    """Battery of simple tests for environments."""

    def test_seed(self, env, env_name):
        seals_test.test_seed(env, env_name, DETERMINISTIC_ENVS)

    def test_premature_step(self, env):
        """Test that you must call reset() before calling step()."""
        seals_test.test_premature_step(
            env,
            skip_fn=pytest.skip,
            raises_fn=pytest.raises,
        )

    def test_model_based(self, env):
        """Smoke test for each of the ModelBasedEnv methods with type checks."""
        if not hasattr(env, "pomdp_state_space"):  # pragma: no cover
            pytest.skip("This test is only for subclasses of ResettableEnv.")

        imitation_test.test_model_based(env)

    def test_rollout_schema(self, env: gym.Env):
        """Tests if environments have correct types on `step()` and `reset()`."""
        seals_test.test_rollout_schema(env)

    def test_render(self, env: gym.Env):
        """Tests `render()` supports modes specified in environment metadata."""
        seals_test.test_render(env, raises_fn=pytest.raises)


def test_dict_extract_wrapper():
    """Tests `DictExtractWrapper` input validation and extraction."""
    venv = vec_env.DummyVecEnv([lambda: envs.SimpleMultiObsEnv()])
    with pytest.raises(KeyError, match="Unrecognized .*"):
        resettable_env.DictExtractWrapper(venv, "foobar")
    wrapped_venv = resettable_env.DictExtractWrapper(venv, "vec")
    with pytest.raises(TypeError, match=".* not dict type"):
        resettable_env.DictExtractWrapper(wrapped_venv, "foobar")
    obs = wrapped_venv.reset()
    assert isinstance(obs, np.ndarray)
    obs, _, _, _ = wrapped_venv.step([wrapped_venv.action_space.sample()])
    assert isinstance(obs, np.ndarray)
