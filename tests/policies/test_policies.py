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


def test_running_norm_identity() -> None:
    """Tests running norm starts and stays at identity function.

    Specifically, we test in evaluation mode (initializatn should not change)
    and in training mode with already normalized data.
    """
    running_norm = base.RunningNorm(1, eps=0.0)
    x = th.Tensor([-1.0, 0.0, 7.32, 42.0])
    running_norm.eval()  # stats should not change in eval mode
    for i in range(10):
        th.testing.assert_equal(running_norm.forward(x), x)
    running_norm.train()  # stats will change in eval mode
    normalized = th.Tensor([-1, 1])  # mean 0, variance 1
    for i in range(10):
        th.testing.assert_equal(running_norm.forward(normalized), normalized)


def test_running_norm_eval_fixed(
    batch_size: int = 8,
    num_batches: int = 10,
    num_features: int = 4,
) -> None:
    """Tests that stats do not change when in eval mode and do when in training."""
    running_norm = base.RunningNorm(num_features)

    def do_forward(shift: float = 0.0, scale: float = 1.0):
        for i in range(num_batches):
            data = th.rand(batch_size, num_features) * scale + shift
            running_norm.forward(data)

    with th.random.fork_rng():
        th.random.manual_seed(42)

        do_forward()
        current_mean = th.clone(running_norm.running_mean)
        current_var = th.clone(running_norm.running_var)

        running_norm.eval()
        do_forward()
        th.testing.assert_equal(running_norm.running_mean, current_mean)
        th.testing.assert_equal(running_norm.running_var, current_var)

        running_norm.train()
        do_forward(1.0, 2.0)
        assert th.all((running_norm.running_mean - current_mean).abs() > 0.01)
        assert th.all((running_norm.running_var - current_var).abs() > 0.01)


@pytest.mark.parametrize("batch_size", [1, 8])
def test_running_norm_matches_dist(batch_size: int) -> None:
    """Test running norm converges to empirical distribution."""
    mean = th.Tensor([-1.3, 0.0, 42])
    var = th.Tensor([0.1, 1.0, 42])
    sd = th.sqrt(var)

    num_dims = len(mean)
    running_norm = base.RunningNorm(num_dims)
    running_norm.train()

    num_samples = 256
    with th.random.fork_rng():
        th.random.manual_seed(42)
        data = th.randn(num_samples, num_dims) * sd + mean
        for start in range(0, num_samples, batch_size):
            batch = data[start : start + batch_size]
            running_norm.forward(batch)

    empirical_mean = th.mean(data, dim=0)
    empirical_var = th.var(data, dim=0, unbiased=False)

    normalized = th.Tensor([[-1.0], [0.0], [1.0], [42.0]])
    normalized = th.tile(normalized, (1, 3))
    scaled = normalized * th.sqrt(empirical_var + running_norm.eps) + empirical_mean
    running_norm.eval()
    for i in range(5):
        th.testing.assert_close(running_norm.forward(scaled), normalized)

    # Stats should match empirical mean (and be unchanged by eval)
    th.testing.assert_close(running_norm.running_mean, empirical_mean)
    th.testing.assert_close(running_norm.running_var, empirical_var)
    assert running_norm.count == num_samples


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
