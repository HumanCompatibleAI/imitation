import logging
import os.path as osp
import pathlib
from typing import Optional, Sequence, Union

import gym
from sacred.observers import FileStorageObserver
from stable_baselines3.common import vec_env
from torch.utils import data as th_data

from imitation.algorithms import bc
from imitation.data import rollout, types
from imitation.scripts.config.train_bc import train_bc_ex
from imitation.util import logger
from imitation.util import sacred as sacred_util


@train_bc_ex.main
def train_bc(
    _run,
    expert_data_src: Union[types.AnyPath, Sequence[types.Trajectory]],
    expert_data_src_format: str,
    observation_space: gym.Space,
    action_space: gym.Space,
    batch_size: int,
    # TODO(shwang): Doesn't currently accept Iterable[Mapping] or
    #  types.TransitionsMinimal, unlike BC.__init__ or BC.set_expert_data_loader().
    n_epochs: Optional[int],
    n_batches: Optional[int],
    n_expert_demos: int,
    l2_weight: float,
    optimizer_kwargs: dict,
    log_dir: types.AnyPath,
    venv: Optional[vec_env.VecEnv],
    log_interval: int,
    log_rollouts_n_episodes: int,
    n_episodes_eval: int,
) -> dict:
    """Sacred interface to Behavioral Cloning.

    Args:
        expert_data_src: Either a path to pickled `Sequence[Trajectory]` or
            `Sequence[Trajectory]`.
        expert_data_src_format: Either "path" if `expert_data_src` is a path, or
            "trajectory" if `expert_data_src` if `Sequence[Trajectory]`.
        observation_space: The observation space corresponding to the expert data.
        action_space: The action space corresponding to the expert data.
        batch_size: Number of observation-action samples used in each BC update.
        n_epochs: The total number of training epochs. Set exactly one of n_epochs and
            n_batches.
        n_batches: The total number of training batches. Set exactly one of n_epochs and
            n_batches.
        log_dir: Log output directory. Final policy is also saved in this directory as
            "{log_dir}/final.pkl"
        venv: If not None, then this VecEnv is used to generate rollout episodes for
            evaluating policy performance during and after training.
        log_interval: The number of updates in between logging various training
            statistics to stdout and Tensorboard.
        log_rollouts_n_episodes: The number of rollout episodes generated for
            training statistics every `log_interval` updates. If `venv` is None or
            this argument is nonpositive, then no rollouts are generated.
        n_episodes_eval: The number of final evaluation rollout episodes, if `venv` is
            provided. These rollouts are used to generate final statistics saved into
            Sacred results, which can be compiled into a table by
            `imitation.scripts.analyze.analyze_imitation`.
    """
    if action_space is None:
        raise ValueError("action_space cannot be None")
    if observation_space is None:
        raise ValueError("observation_space cannot be None")

    log_dir = pathlib.Path(log_dir)
    log_dir.mkdir(parents=True, exist_ok=True)
    logging.info("Logging to %s", log_dir)

    logger.configure(log_dir, ["tensorboard", "stdout"])
    sacred_util.build_sacred_symlink(log_dir, _run)

    if expert_data_src_format == "path":
        expert_trajs = types.load(expert_data_src)
    elif expert_data_src_format == "trajectory":
        # Convenience option for launching experiment from Python script with
        # in-memory trajectories.
        expert_trajs = expert_data_src
    else:
        raise ValueError(f"Invalid expert_data_src_format={expert_data_src_format}")

    # TODO(shwang): Copied from scripts/train_adversarial -- refactor with "auto",
    # or combine all train_*.py into a single script?
    if n_expert_demos is not None:
        if not len(expert_trajs) >= n_expert_demos:
            raise ValueError(
                f"Want to use n_expert_demos={n_expert_demos} trajectories, but only "
                f"{len(expert_trajs)} are available."
            )
        expert_trajs = expert_trajs[:n_expert_demos]

    expert_data_trans = rollout.flatten_trajectories(expert_trajs)
    expert_data = th_data.DataLoader(
        expert_data_trans,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=types.transitions_collate_fn,
    )

    model = bc.BC(
        observation_space,
        action_space,
        expert_data=expert_data,
        l2_weight=l2_weight,
        optimizer_kwargs=optimizer_kwargs,
    )
    model.train(
        n_epochs=n_epochs,
        n_batches=n_batches,
        log_interval=log_interval,
        log_rollouts_venv=venv,
        log_rollouts_n_episodes=log_rollouts_n_episodes,
    )
    model.save_policy(policy_path=pathlib.Path(log_dir, "final.th"))

    print(f"Visualize results with: tensorboard --logdir '{log_dir}'")

    # TODO(shwang): Use auto env, auto stats thing with shared `env` and stats
    #  ingredient, or something like that.
    sample_until = rollout.make_sample_until(
        n_timesteps=None, n_episodes=n_episodes_eval
    )
    trajs = rollout.generate_trajectories(
        model.policy,
        venv,
        sample_until=sample_until,
    )
    results = {}
    results["expert_stats"] = rollout.rollout_stats(expert_trajs)
    results["imit_stats"] = rollout.rollout_stats(trajs)
    return results


def main_console():
    observer = FileStorageObserver(osp.join("output", "sacred", "train_bc"))
    train_bc_ex.observers.append(observer)
    train_bc_ex.run_commandline()


if __name__ == "__main__":
    main_console()
