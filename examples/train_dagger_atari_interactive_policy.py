"""Training DAgger with an interactive policy that queries the user for actions.

Note that this is a toy example that does not lead to training a reasonable policy.
"""

import tempfile

import gymnasium as gym
import numpy as np
import torch as th
from stable_baselines3.common import torch_layers, vec_env

from imitation.algorithms import bc, dagger
from imitation.data import wrappers as data_wrappers
from imitation.policies import interactive
from imitation.policies import wrappers as policy_wrappers
from imitation.policies import base as policy_base


if __name__ == "__main__":
    rng = np.random.default_rng(0)

    env = gym.make("PongNoFrameskip-v4", render_mode="rgb_array")
    hr_env = data_wrappers.HumanReadableWrapper(env)
    venv = vec_env.DummyVecEnv([lambda: hr_env])
    venv.seed(0)

    expert = interactive.AtariInteractivePolicy(venv)
    policy = policy_base.FeedForward32Policy(
        observation_space=env.observation_space,
        action_space=env.action_space,
        # Set lr_schedule to max value to force error if policy.optimizer
        # is used by mistake (should use self.optimizer instead).
        lr_schedule=lambda _: th.finfo(th.float32).max,
        features_extractor_class=torch_layers.FlattenExtractor,
    )
    wrapped_policy = policy_wrappers.ExcludeHRWrapper(policy)

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        policy=wrapped_policy,
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
