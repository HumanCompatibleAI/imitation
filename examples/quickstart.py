"""Loads CartPole-v1 demonstrations and trains BC, GAIL, and AIRL models on that data.
"""

import pathlib
import pickle
import tempfile

import stable_baselines3 as sb3

from imitation.algorithms import adversarial, bc
from imitation.data import rollout
from imitation.util import logger, util

# Load pickled test demonstrations.
with open("tests/data/expert_models/cartpole_0/rollouts/final.pkl", "rb") as f:
    # This is a list of `imitation.data.types.Trajectory`, where
    # every instance contains observations and actions for a single expert
    # demonstration.
    trajectories = pickle.load(f)

# Convert List[types.Trajectory] to an instance of `imitation.data.types.Transitions`.
# This is a more general dataclass containing unordered
# (observation, actions, next_observation) transitions.
transitions = rollout.flatten_trajectories(trajectories)

venv = util.make_vec_env("CartPole-v1", n_envs=2)

tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
tempdir_path = pathlib.Path(tempdir.name)
print(f"All Tensorboards and logging are being written inside {tempdir_path}/.")

# Train BC on expert data.
# BC also accepts as `expert_data` any PyTorch-style DataLoader that iterates over
# dictionaries containing observations and actions.
logger.configure(tempdir_path / "BC/")
bc_trainer = bc.BC(venv.observation_space, venv.action_space, expert_data=transitions)
bc_trainer.train(n_epochs=1)

# Train GAIL on expert data.
# GAIL, and AIRL also accept as `expert_data` any Pytorch-style DataLoader that
# iterates over dictionaries containing observations, actions, and next_observations.
logger.configure(tempdir_path / "GAIL/")
gail_trainer = adversarial.GAIL(
    venv,
    expert_data=transitions,
    expert_batch_size=32,
    gen_algo=sb3.PPO("MlpPolicy", venv, verbose=1, n_steps=1024),
)
gail_trainer.train(total_timesteps=2048)

# Train AIRL on expert data.
logger.configure(tempdir_path / "AIRL/")
airl_trainer = adversarial.AIRL(
    venv,
    expert_data=transitions,
    expert_batch_size=32,
    gen_algo=sb3.PPO("MlpPolicy", venv, verbose=1, n_steps=1024),
)
airl_trainer.train(total_timesteps=2048)
