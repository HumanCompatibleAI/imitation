"""Trains BC, GAIL and AIRL models on saved CartPole-v1 demonstrations."""

import pathlib
import pickle
import tempfile

import seals  # noqa: F401
import stable_baselines3 as sb3

from imitation.algorithms import bc
from imitation.algorithms.adversarial import airl, gail
from imitation.data import rollout
from imitation.util import logger, util

# Load pickled test demonstrations.
with open("tests/testdata/expert_models/cartpole_0/rollouts/final.pkl", "rb") as f:
    # This is a list of `imitation.data.types.Trajectory`, where
    # every instance contains observations and actions for a single expert
    # demonstration.
    trajectories = pickle.load(f)

# Convert List[types.Trajectory] to an instance of `imitation.data.types.Transitions`.
# This is a more general dataclass containing unordered
# (observation, actions, next_observation) transitions.
transitions = rollout.flatten_trajectories(trajectories)

venv = util.make_vec_env("seals/CartPole-v0", n_envs=2)

tempdir = tempfile.TemporaryDirectory(prefix="quickstart")
tempdir_path = pathlib.Path(tempdir.name)
print(f"All Tensorboards and logging are being written inside {tempdir_path}/.")

# Train BC on expert data.
# BC also accepts as `demonstrations` any PyTorch-style DataLoader that iterates over
# dictionaries containing observations and actions.
bc_logger = logger.configure(tempdir_path / "BC/")
bc_trainer = bc.BC(
    observation_space=venv.observation_space,
    action_space=venv.action_space,
    demonstrations=transitions,
    custom_logger=bc_logger,
)
bc_trainer.train(n_epochs=1)

# Train GAIL on expert data.
# GAIL, and AIRL also accept as `demonstrations` any Pytorch-style DataLoader that
# iterates over dictionaries containing observations, actions, and next_observations.
gail_logger = logger.configure(tempdir_path / "GAIL/")
gail_trainer = gail.GAIL(
    venv=venv,
    demonstrations=transitions,
    demo_batch_size=32,
    gen_algo=sb3.PPO("MlpPolicy", venv, verbose=1, n_steps=1024),
    custom_logger=gail_logger,
)
gail_trainer.train(total_timesteps=2048)

# Train AIRL on expert data.
airl_logger = logger.configure(tempdir_path / "AIRL/")
airl_trainer = airl.AIRL(
    venv=venv,
    demonstrations=transitions,
    demo_batch_size=32,
    gen_algo=sb3.PPO("MlpPolicy", venv, verbose=1, n_steps=1024),
    custom_logger=airl_logger,
)
airl_trainer.train(total_timesteps=2048)
