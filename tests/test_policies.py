"""Test imitation.policies."""

import tempfile

import gym
import numpy as np
import pytest
from stable_baselines.common.vec_env import VecNormalize

from imitation.policies import serialize
from imitation.util import rollout, util

SIMPLE_ENVS = [
    "CartPole-v0",  # Discrete(2) action space
    "MountainCarContinuous-v0",  # Box(1) action space
]
HARDCODED_TYPES = ["random", "zero"]
BASELINE_MODELS = [(name, cls)
                   for name, (cls, attr) in
                   serialize.STABLE_BASELINES_CLASSES.items()]


@pytest.mark.parametrize("env_name", SIMPLE_ENVS)
@pytest.mark.parametrize("policy_type", HARDCODED_TYPES)
def test_actions_valid(env_name, policy_type):
  """Test output actions of our custom policies always lie in action space."""
  env = gym.make(env_name)
  policy = serialize.load_policy(policy_type, "foobar", env)
  transitions = rollout.generate_transitions(policy, env, n_timesteps=100)
  old_obs, act, new_obs, rew = transitions

  for a in act:
    assert env.action_space.contains(a)


@pytest.mark.parametrize("env_name", SIMPLE_ENVS)
@pytest.mark.parametrize("model_cfg", BASELINE_MODELS)
@pytest.mark.parametrize("normalize", [False, True])
def test_serialize_identity(env_name, model_cfg, normalize):
  """Test output actions of deserialized policy are same as original."""
  orig_venv = venv = util.make_vec_env(env_name, n_envs=1, parallel=False)
  vec_normalize = None
  if normalize:
    venv = vec_normalize = VecNormalize(venv)
  model_name, model_cls = model_cfg
  model = model_cls('MlpPolicy', venv)
  model.learn(1000)

  venv.env_method('seed', 0)
  venv.reset()
  if normalize:
    # don't want statistics to change as we collect rollouts
    vec_normalize.training = False
  orig_rollout = rollout.generate_transitions(model, venv, n_timesteps=1000,
                                              deterministic_policy=True)

  with tempfile.TemporaryDirectory(prefix='imitation-serialize-pol') as tmpdir:
    serialize.save_stable_model(tmpdir, model, vec_normalize)
    loaded = serialize.load_policy(model_name, tmpdir, orig_venv)

  orig_venv.env_method('seed', 0)
  orig_venv.reset()
  new_rollout = rollout.generate_transitions(loaded, orig_venv,
                                             n_timesteps=1000,
                                             deterministic_policy=True)

  orig_acts = orig_rollout[1]
  new_acts = new_rollout[1]
  assert np.allclose(orig_acts, new_acts)
