"""Tests `imitation.policies.*`."""

import functools
import pathlib

import gym
import numpy as np
import pytest
import torch as th
from stable_baselines3.common import preprocessing
from torch import nn

from imitation.data import rollout
from imitation.policies import base, serialize
from imitation.util import registry, util

SIMPLE_DISCRETE_ENV = "CartPole-v0"  # Discrete(2) action space
SIMPLE_CONTINUOUS_ENV = "MountainCarContinuous-v0"  # Box(1) action space
SIMPLE_ENVS = [SIMPLE_DISCRETE_ENV, SIMPLE_CONTINUOUS_ENV]
HARDCODED_TYPES = ["random", "zero"]

assert_equal = functools.partial(th.testing.assert_close, rtol=0, atol=0)


@pytest.mark.parametrize("env_name", SIMPLE_ENVS)
@pytest.mark.parametrize("policy_type", HARDCODED_TYPES)
def test_actions_valid(env_name, policy_type):
    """Test output actions of our custom policies always lie in action space."""
    venv = util.make_vec_env(env_name, n_envs=1, parallel=False)
    policy = serialize.load_policy(policy_type, "foobar", venv)
    transitions = rollout.generate_transitions(policy, venv, n_timesteps=100)

    for a in transitions.acts:
        assert venv.action_space.contains(a)


@pytest.mark.parametrize(
    "policy_env_name_pair",
    [
        ("ppo", SIMPLE_DISCRETE_ENV),
        ("sac", SIMPLE_CONTINUOUS_ENV),
    ],
)
def test_save_stable_model_errors_and_warnings(tmpdir, policy_env_name_pair):
    """Check errors and warnings in `save_stable_model()`."""
    policy, env_name = policy_env_name_pair
    tmpdir = pathlib.Path(tmpdir)
    venv = util.make_vec_env(env_name)

    # Trigger FileNotFoundError for no model.{zip,pkl}
    dir_a = tmpdir / "a"
    dir_a.mkdir()
    with pytest.raises(FileNotFoundError, match=".*No such file or.*model.zip.*"):
        serialize.load_policy(policy, str(dir_a), venv)

    vec_normalize = dir_a / "vec_normalize.pkl"
    vec_normalize.touch()
    with pytest.raises(FileExistsError, match="Outdated policy format.*"):
        serialize.load_policy(policy, str(dir_a), venv)

    # Trigger FileNotError for nonexistent directory
    dir_nonexistent = tmpdir / "i_dont_exist"
    with pytest.raises(FileNotFoundError, match=".*needs to be a directory.*"):
        serialize.load_policy(policy, str(dir_nonexistent), venv)


def _test_serialize_identity(env_name, model_cfg, tmpdir):
    """Test output actions of deserialized policy are same as original."""
    venv = util.make_vec_env(env_name, n_envs=1, parallel=False)

    model_name, model_cls_name = model_cfg
    model_cls = registry.load_attr(model_cls_name)

    model = model_cls("MlpPolicy", venv)
    model.learn(1000)

    venv.env_method("seed", 0)
    venv.reset()
    orig_rollout = rollout.generate_transitions(
        model,
        venv,
        n_timesteps=1000,
        deterministic_policy=True,
        rng=np.random.RandomState(0),
    )

    serialize.save_stable_model(tmpdir, model)
    loaded = serialize.load_policy(model_name, tmpdir, venv)
    venv.env_method("seed", 0)
    venv.reset()
    new_rollout = rollout.generate_transitions(
        loaded,
        venv,
        n_timesteps=1000,
        deterministic_policy=True,
        rng=np.random.RandomState(0),
    )

    assert np.allclose(orig_rollout.acts, new_rollout.acts)


SB_CONFIGS = serialize.STABLE_BASELINES_CLASSES.items()
CONTINUOUS_ONLY = ["sac"]
NORMAL_CONFIGS = [cfg for cfg in SB_CONFIGS if cfg[0] not in CONTINUOUS_ONLY]
CONTINUOUS_ONLY_CONFIGS = [cfg for cfg in SB_CONFIGS if cfg[0] in CONTINUOUS_ONLY]


@pytest.mark.parametrize("env_name", SIMPLE_ENVS)
@pytest.mark.parametrize("model_cfg", NORMAL_CONFIGS)
def test_serialize_identity(env_name, model_cfg, tmpdir):
    """Test output actions of deserialized policy are same as original."""
    _test_serialize_identity(env_name, model_cfg, tmpdir)


@pytest.mark.parametrize("env_name", [SIMPLE_CONTINUOUS_ENV])
@pytest.mark.parametrize("model_cfg", CONTINUOUS_ONLY_CONFIGS)
def test_serialize_identity_continuous_only(env_name, model_cfg, tmpdir):
    """Test serialize identity for continuous_only algorithms."""
    _test_serialize_identity(env_name, model_cfg, tmpdir)


class ZeroModule(nn.Module):
    """Module that always returns zeros of same shape as input."""

    def __init__(self, features_dim: int):
        """Builds ZeroModule."""
        super().__init__()
        self.features_dim = features_dim

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Returns zeros of same shape as `x`."""
        assert x.shape[1:] == (self.features_dim,)
        return th.zeros_like(x)


@pytest.mark.parametrize(
    "obs_space",
    [
        gym.spaces.Box(-1, 1, shape=(1,)),
        gym.spaces.Box(-1, 1, shape=(3,)),
        gym.spaces.Box(-1, 1, shape=(3, 4)),
        gym.spaces.Discrete(2),
        gym.spaces.MultiDiscrete([5, 2]),
    ],
)
def test_normalize_features_extractor(obs_space: gym.Space) -> None:
    """Tests `base.NormalizeFeaturesExtractor`.

    Verifies it returns features of appropriate shape (based on flattening `obs_space).
    Also checks that using an identity normalizer leaves observations unchanged,
    and that using `ZeroModule` returns all zeros.

    Args:
        obs_space: The observation space to sample from.
    """
    extractors = {
        "norm": base.NormalizeFeaturesExtractor(obs_space),
        "identity": base.NormalizeFeaturesExtractor(obs_space, nn.Identity),
        "zero": base.NormalizeFeaturesExtractor(obs_space, ZeroModule),
    }

    for i in range(10):
        obs = th.as_tensor([obs_space.sample()])
        obs = preprocessing.preprocess_obs(obs, obs_space)
        flattened_obs = obs.flatten(1, -1)
        extracted = {k: extractor(obs) for k, extractor in extractors.items()}
        for k, v in extracted.items():
            assert v.shape == flattened_obs.shape, k

        assert_equal(extracted["identity"], flattened_obs)
        assert_equal(extracted["zero"], flattened_obs * 0.0)
