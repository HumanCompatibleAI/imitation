"""This script demonstrates how to train using DAgger and interactive demonstrations.

The script uses Atari Pong.
"""
import tempfile

import numpy as np
from stable_baselines3.common.evaluation import evaluate_policy

from imitation.algorithms import bc
from imitation.algorithms.dagger import SimpleDAggerTrainer
from imitation.policies.interactive import InteractivePolicy
from imitation.util.util import make_vec_env

# todo: also test with gym.env; # todo: merge with train_dagger_with_human_demos.py
env = make_vec_env(
    "ALE/Pong-v5",
    n_envs=1,
    rng=np.random.default_rng(42),
    env_make_kwargs=dict(render_mode="human"),
)
expert = InteractivePolicy(env)

bc_trainer = bc.BC(
    observation_space=env.observation_space,
    action_space=env.action_space,
    rng=np.random.default_rng(),
)

tmpdir = tempfile.mkdtemp(prefix="dagger_human_example_")
print(tmpdir)
dagger_trainer = SimpleDAggerTrainer(
    venv=env,
    scratch_dir=tmpdir,
    expert_policy=expert,
    bc_trainer=bc_trainer,
    rng=np.random.default_rng(),
)

reward_before, _ = evaluate_policy(dagger_trainer.policy, env, 20)
print(f"{reward_before=}")


dagger_trainer.train(
    total_timesteps=10,
    rollout_round_min_timesteps=10,
)

reward_after, _ = evaluate_policy(dagger_trainer.policy, env, 20)
print(f"{reward_after=}")
