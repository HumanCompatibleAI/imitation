"""Training DAgger with an interactive policy that queries the user for actions.

Note that this is a toy example that does not lead to training a reasonable policy.
"""

import tempfile

import gymnasium as gym
import numpy as np
from stable_baselines3.common import vec_env

from imitation.algorithms import bc, dagger
from imitation.data import wrappers
from imitation.policies import interactive

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")
    env = wrappers.HumanReadableWrapper(env)
    venv = vec_env.DummyVecEnv([lambda: env])
    venv.seed(0)

    expert = interactive.AtariInteractivePolicy(venv)

    venv_with_no_rgb = wrappers.RemoveHumanReadableWrapper(venv)
    bc_trainer = bc.BC(
        observation_space=venv_with_no_rgb.observation_space,
        action_space=venv_with_no_rgb.action_space,
        rng=rng,
    )

    with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
        dagger_trainer = dagger.SimpleDAggerTrainer(
            venv=venv,
            scratch_dir=tmpdir,
            expert_policy=expert,
            bc_trainer=bc_trainer,
            rng=rng,
        )
        dagger_trainer.train(
            total_timesteps=20,
            rollout_round_min_episodes=1,
            rollout_round_min_timesteps=10,
        )
