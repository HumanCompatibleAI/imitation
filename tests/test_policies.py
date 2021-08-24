"""Test imitation.policies."""

import pathlib

import numpy as np
import pytest
import stable_baselines3
from stable_baselines3.common.vec_env import VecNormalize

from imitation.data import rollout
from imitation.policies import serialize
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
    """Check errors and warnings in save_stable_model()"""
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

    # FIXME(sam): verbose=1 is a hack to stop it from setting up SB logger
    model = model_cls("MlpPolicy", venv, verbose=1)
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
