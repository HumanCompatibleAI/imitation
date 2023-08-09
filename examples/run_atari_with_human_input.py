import tempfile

import gym
import numpy as np
import retro
from retro import RetroEnv
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import DummyVecEnv

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.policies.interactive import InteractivePolicy
from imitation.util.util import make_vec_env

# todo: also test with gym.env; # todo: merge with train_dagger_with_human_demos.py
# env = retro.make("ALE/Pong-v5")  # FileNotFoundError: No romfiles found for game...
env = retro.make("Pong-Atari2600")

# env = make_vec_env(
#     "ALE/Pong-v5",
#     n_envs=1,
#     rng=np.random.default_rng(42),
#     env_make_kwargs=dict(render_mode="human"),
#     #     def __init__(self, game,
#     #     state=retro.State.DEFAULT,
#     #     scenario=None, info=None,
#     #     use_restricted_actions=retro.Actions.FILTERED,
#     #     record=False, players=1,
#     #     inttype=retro.data.Integrations.STABLE,
#     #     obs_type=retro.Observations.IMAGE
#     #     ):
#     post_wrappers=[lambda env, _: RetroEnv(env)],
# )
assert isinstance(env, RetroEnv)
assert isinstance(env, gym.Env)

# wrap in VecEnv
venv = DummyVecEnv([lambda: env])
env
venv


expert = InteractivePolicy(venv)
