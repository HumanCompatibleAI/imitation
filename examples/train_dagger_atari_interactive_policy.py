import tempfile

import gym
import numpy as np
from stable_baselines3.common import vec_env

from imitation.algorithms import bc, dagger
from imitation.policies import interactive

if __name__ == "__main__":
    rng = np.random.default_rng(0)

    env = vec_env.DummyVecEnv([lambda: gym.wrappers.TimeLimit(gym.make("Pong-v4"), 10)])
    env.seed(0)

    action_names = env.envs[0].get_action_meanings()
    names_to_keys = {
        "NOOP": "n",
        "FIRE": "f",
        "LEFT": "w",
        "RIGHT": "e",
        "LEFTFIRE": "q",
        "RIGHTFIRE": "r",
    }
    action_keys = list(map(names_to_keys.get, action_names))

    expert = interactive.ImageObsDiscreteInteractivePolicy(
        env.observation_space, env.action_space, action_names, action_keys
    )

    bc_trainer = bc.BC(
        observation_space=env.observation_space,
        action_space=env.action_space,
        rng=rng,
    )

    with tempfile.TemporaryDirectory(prefix="dagger_example_") as tmpdir:
        dagger_trainer = dagger.SimpleDAggerTrainer(
            venv=env,
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
