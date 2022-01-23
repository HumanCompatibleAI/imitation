"""Tests `imitation.policies.*`."""

import pathlib

import gym
import numpy as np
import pytest
import stable_baselines3
import torch as th
from stable_baselines3.common import preprocessing
from stable_baselines3.common.vec_env import VecNormalize
from torch import nn

from imitation.data import rollout
from imitation.policies import base, serialize
from imitation.util import registry, util

SIMPLE_ENVS = [
    "CartPole-v0",  # Discrete(2) action space
    "MountainCarContinuous-v0",  # Box(1) action space
]
HARDCODED_TYPES = ["random", "zero"]
BASELINE_MODELS = [
    (name, cls_name)
    for name, (cls_name, attr) in serialize.STABLE_BASELINES_CLASSES.items()
]


@pytest.mark.parametrize("env_name", SIMPLE_ENVS)
@pytest.mark.parametrize("policy_type", HARDCODED_TYPES)
def test_actions_valid(env_name, policy_type):
    """Test output actions of our custom policies always lie in action space."""
    venv = util.make_vec_env(env_name, n_envs=1, parallel=False)
    policy = serialize.load_policy(policy_type, "foobar", venv)
    transitions = rollout.generate_transitions(policy, venv, n_timesteps=100)

    for a in transitions.acts:
        assert venv.action_space.contains(a)


def test_save_stable_model_errors_and_warnings(tmpdir):
    """Check errors and warnings in `save_stable_model()`."""
    tmpdir = pathlib.Path(tmpdir)
    venv = util.make_vec_env("CartPole-v0")
    ppo = stable_baselines3.PPO("MlpPolicy", venv)

    # Trigger DeprecationWarning for saving to model.pkl instead of model.zip
    dir_a = tmpdir / "a"
    dir_a.mkdir()
    deprecated_model_path = dir_a / "model.pkl"
    ppo.save(deprecated_model_path)
    with pytest.warns(DeprecationWarning, match=".*deprecated policy directory.*"):
        serialize.load_policy("ppo", str(dir_a), venv)

    # Trigger FileNotFoundError for no model.{zip,pkl}
    dir_b = tmpdir / "b"
    dir_b.mkdir()
    with pytest.raises(FileNotFoundError, match=".*Could not find.*model.zip.*"):
        serialize.load_policy("ppo", str(dir_b), venv)

    # Trigger FileNotError for nonexistent directory
    dir_nonexistent = tmpdir / "i_dont_exist"
    with pytest.raises(FileNotFoundError, match=".*needs to be a directory.*"):
        serialize.load_policy("ppo", str(dir_nonexistent), venv)


@pytest.mark.parametrize("env_name", SIMPLE_ENVS)
@pytest.mark.parametrize("model_cfg", BASELINE_MODELS)
@pytest.mark.parametrize("normalize", [False, True])
def test_serialize_identity(env_name, model_cfg, normalize, tmpdir):
    """Test output actions of deserialized policy are same as original."""
    orig_venv = venv = util.make_vec_env(env_name, n_envs=1, parallel=False)
    vec_normalize = None
    if normalize:
        venv = vec_normalize = VecNormalize(venv)

    model_name, model_cls_name = model_cfg
    model_cls = registry.load_attr(model_cls_name)

    model = model_cls("MlpPolicy", venv)
    model.learn(1000)

    venv.env_method("seed", 0)
    venv.reset()
    if normalize:
        # don't want statistics to change as we collect rollouts
        vec_normalize.training = False
    orig_rollout = rollout.generate_transitions(
        model,
        venv,
        n_timesteps=1000,
        deterministic_policy=True,
        rng=np.random.RandomState(0),
    )

    serialize.save_stable_model(tmpdir, model, vec_normalize)
    # We use `orig_venv` since `load_policy` automatically wraps `loaded`
    # with a VecNormalize, when appropriate.
    loaded = serialize.load_policy(model_name, tmpdir, orig_venv)
    orig_venv.env_method("seed", 0)
    orig_venv.reset()
    new_rollout = rollout.generate_transitions(
        loaded,
        orig_venv,
        n_timesteps=1000,
        deterministic_policy=True,
        rng=np.random.RandomState(0),
    )

    assert np.allclose(orig_rollout.acts, new_rollout.acts)


class ZeroModule(nn.Module):
    """Module that always returns zeros of same shape as input."""

    def __init__(self, features_dim: int):
        """Builds ZeroModule."""
        super().__init__()
        self.features_dim = features_dim

    def forward(self, x: th.Tensor) -> th.Tensor:
        """Returns zeros of same shape as `x`."""
        assert x.shape[1] == self.features_dim
        return x * 0


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

        th.testing.assert_equal(extracted["identity"], flattened_obs)
        th.testing.assert_equal(extracted["zero"], flattened_obs * 0.0)
