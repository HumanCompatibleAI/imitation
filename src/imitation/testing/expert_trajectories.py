"""Test utilities to conveniently generate expert trajectories."""
import math
import pathlib
import pickle
import shutil
import warnings
from os import PathLike
from pathlib import Path
from typing import Sequence

import huggingface_sb3 as hfsb3
import numpy as np
from filelock import FileLock
from torch.utils import data as th_data

import imitation.data.serialize as data_serialize
import imitation.policies.serialize as policies_serialize
from imitation.data import rollout, types, wrappers
from imitation.util import util


def generate_expert_trajectories(
    env_id: str,
    num_trajectories: int,
    rng: np.random.Generator,
) -> Sequence[types.TrajectoryWithRew]:  # pragma: no cover
    """Generate expert trajectories for the given environment.

    Note: will just pull a pretrained policy from the Hugging Face model hub.

    Args:
        env_id: The environment to generate trajectories for.
        num_trajectories: The number of trajectories to generate.
        rng: The random number generator to use.

    Returns:
        A list of trajectories with rewards.
    """
    env = util.make_vec_env(
        env_id,
        post_wrappers=[lambda e, _: wrappers.RolloutInfoWrapper(e)],
        rng=rng,
    )
    try:
        expert = policies_serialize.load_policy("ppo-huggingface", env, env_name=env_id)
        return rollout.rollout(
            expert,
            env,
            rollout.make_sample_until(min_episodes=num_trajectories),
            rng=rng,
        )
    finally:
        env.close()


def lazy_generate_expert_trajectories(
    cache_path: PathLike,
    env_id: str,
    num_trajectories: int,
    rng: np.random.Generator,
) -> Sequence[types.TrajectoryWithRew]:
    """Generate or load expert trajectories from cache.

    Args:
        cache_path: A path to the folder to be used as cache for the expert
            trajectories.
        env_id: The environment to generate trajectories for.
        num_trajectories: The number of trajectories to generate.
        rng: The random number generator to use.

    Returns:
        A list of trajectories with rewards.
    """
    environment_cache_path = pathlib.Path(cache_path) / hfsb3.EnvironmentName(env_id)
    environment_cache_path.mkdir(parents=True, exist_ok=True)

    trajectories_path = environment_cache_path / "rollout"

    # Note: we cast to str here because FileLock doesn't support pathlib.Path.
    with FileLock(str(environment_cache_path / "rollout.lock")):
        try:
            trajectories = data_serialize.load_with_rewards(trajectories_path)
        except (FileNotFoundError, pickle.PickleError) as e:  # pragma: no cover
            generation_reason = (
                "the cache is cold"
                if isinstance(e, FileNotFoundError)
                else "trajectory file format in the cache is outdated"
            )
            warnings.warn(
                f"Generating expert trajectories for {env_id} because "
                f"{generation_reason}.",
            )
            trajectories = generate_expert_trajectories(env_id, num_trajectories, rng)
            data_serialize.save(trajectories_path, trajectories)

    if len(trajectories) >= num_trajectories:
        return trajectories[:num_trajectories]
    else:  # pragma: no cover
        # If it is not enough, just throw away the cache and generate more.
        if trajectories_path.is_dir():
            # rmtree won't remove directory on Windows until the last handle to the directory is closed
            del trajectories
            shutil.rmtree(trajectories_path)
        else:
            trajectories_path.unlink()
        return lazy_generate_expert_trajectories(
            cache_path,
            env_id,
            num_trajectories,
            rng,
        )


def make_expert_transition_loader(
    cache_dir: Path,
    batch_size: int,
    expert_data_type: str,
    env_name: str,
    rng: np.random.Generator,
    num_trajectories: int = 1,
):
    """Creates different kinds of PyTorch data loaders for expert transitions.

    Args:
        cache_dir: The directory to use for caching the expert trajectories.
        batch_size: The batch size to use for the data loader.
        expert_data_type: The type of expert data to use. Can be one of "data_loader",
            "ducktyped_data_loader", "transitions".
        env_name: The environment to generate trajectories for.
        rng: The random number generator to use.
        num_trajectories: The number of trajectories to generate.

    Raises:
        ValueError: If `expert_data_type` is not one of the supported types.

    Returns:
        A pytorch data loader for expert transitions.
    """
    trajectories = lazy_generate_expert_trajectories(
        cache_dir,
        env_name,
        num_trajectories,
        rng,
    )
    transitions = rollout.flatten_trajectories(trajectories)

    if len(transitions) < batch_size:  # pragma: no cover
        # If we have less transitions than the batch size, we estimate the trajectory
        # length and generate enough trajectories to fill the batch size.
        trajectory_length = len(transitions) // len(trajectories)
        min_required_trajectories = math.ceil(batch_size / trajectory_length)
        transitions = rollout.flatten_trajectories(
            lazy_generate_expert_trajectories(
                cache_dir,
                env_name,
                min_required_trajectories,
                rng,
            ),
        )

    if expert_data_type == "data_loader":
        return th_data.DataLoader(
            transitions,
            batch_size=batch_size,
            shuffle=True,
            drop_last=True,
            collate_fn=types.transitions_collate_fn,
        )
    elif expert_data_type == "ducktyped_data_loader":

        class DucktypedDataset:
            """Used to check that any iterator over Dict[str, Tensor] works with BC."""

            def __init__(self, transitions: types.TransitionsMinimal, batch_size: int):
                """Builds `DucktypedDataset`."""
                self.trans = transitions
                self.batch_size = batch_size

            def __iter__(self):
                for start in range(
                    0,
                    len(self.trans) - self.batch_size,
                    self.batch_size,
                ):
                    end = start + self.batch_size
                    d = dict(
                        obs=self.trans.obs[start:end],
                        acts=self.trans.acts[start:end],
                    )
                    d = {k: util.safe_to_tensor(v) for k, v in d.items()}
                    yield d

        return DucktypedDataset(transitions, batch_size)
    elif expert_data_type == "transitions":
        return transitions
    else:  # pragma: no cover
        raise ValueError(f"Unexpected data type '{expert_data_type}'")
