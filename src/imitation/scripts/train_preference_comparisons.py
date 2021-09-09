"""Train a reward model using preference comparisons.

Can be used as a CLI script, or the `train_preference_comparisons` function
can be called directly.
"""

import os
from typing import Any, Dict, Optional

import stable_baselines3
import torch as th
from sacred.observers import FileStorageObserver
from stable_baselines3.common import vec_env

from imitation.algorithms import preference_comparisons
from imitation.data import rollout
from imitation.policies import serialize, trainer
from imitation.rewards import reward_nets
from imitation.scripts.config.train_preference_comparisons import (
    train_preference_comparisons_ex,
)
from imitation.util import logger
from imitation.util import sacred as sacred_util
from imitation.util import util


@train_preference_comparisons_ex.main
def train_preference_comparisons(
    _run,
    _seed: int,
    env_name: str,
    num_vec: int,
    parallel: bool,
    normalize: bool,
    normalize_kwargs: dict,
    max_episode_steps: Optional[int],
    log_dir: str,
    total_timesteps: int,
    total_comparisons: int,
    comparisons_per_iteration: int,
    fragment_length: int,
    transition_oversampling: float,
    n_episodes_eval: int,
    trajectory_path: Optional[str],
    preferences_path: Optional[str],
    save_preferences: bool,
    agent_path: Optional[str],
    reward_net_kwargs: Dict[str, Any],
    reward_trainer_kwargs: Dict[str, Any],
    agent_kwargs: Dict[str, Any],
    gatherer_kwargs: Dict[str, Any],
    allow_variable_horizon: bool,
) -> dict:
    """Train a reward model using preference comparisons.

    Args:
        _seed: Random seed.
        env_name: The environment to train in.
        num_vec: Number of `gym.Env` to vectorize.
        parallel: Whether to use "true" parallelism. If True, then use `SubProcVecEnv`.
            Otherwise, use `DummyVecEnv` which steps through environments serially.
        normalize: If True, then rescale observations and reward.
        normalize_kwargs: kwargs for `VecNormalize`.
        max_episode_steps: If not None, then a TimeLimit wrapper is applied to each
            environment to artificially limit the maximum number of timesteps in an
            episode.
        log_dir: Directory to save models and other logging to.
        total_timesteps: number of environment interaction steps
        total_comparisons: number of preferences to gather in total
        comparisons_per_iteration: number of preferences to gather at once (before
            switching back to agent training). This doesn't impact the total number
            of comparisons that are gathered, only the frequency of switching
            between preference gathering and agent training.
        fragment_length: number of timesteps per fragment that is used to elicit
            preferences
        transition_oversampling: factor by which to oversample transitions before
            creating fragments. Since fragments are sampled with replacement,
            this is usually chosen > 1 to avoid having the same transition
            in too many fragments.
        n_episodes_eval: The number of episodes to average over when calculating
            the average episode reward of the learned policy for return.
        trajectory_path: either None, in which case an agent will be trained
            and used to sample trajectories on the fly, or a path to a pickled
            sequence of TrajectoryWithRew to be trained on
        preferences_path: path to a dataset of preferences previously created
            using this script. If given, only a reward model will be trained
            on this preference dataset, no agent will be trained and no new
            preferences will be gathered.
        save_preferences: if True, store the final dataset of preferences to disk.
        agent_path: if given, initialize the agent using this stored policy
            rather than randomly.
        reward_net_kwargs: passed to BasicRewardNet
        reward_trainer_kwargs: passed to CrossEntropyRewardTrainer
        agent_kwargs: passed to SB3's PPO
        gatherer_kwargs: passed to SyntheticGatherer
        allow_variable_horizon: If False (default), algorithm will raise an
            exception if it detects trajectories of different length during
            training. If True, overrides this safety check. WARNING: variable
            horizon episodes leak information about the reward via termination
            condition, and can seriously confound evaluation. Read
            https://imitation.readthedocs.io/en/latest/guide/variable_horizon.html
            before overriding this.
    """

    custom_logger = logger.configure(log_dir, ["tensorboard", "stdout"])
    os.makedirs(log_dir, exist_ok=True)
    sacred_util.build_sacred_symlink(log_dir, _run)

    venv = util.make_vec_env(
        env_name,
        num_vec,
        seed=_seed,
        parallel=parallel,
        log_dir=log_dir,
        max_episode_steps=max_episode_steps,
    )

    vec_normalize = None
    if normalize:
        venv = vec_normalize = vec_env.VecNormalize(venv, **normalize_kwargs)

    reward_net = reward_nets.BasicRewardNet(
        venv.observation_space, venv.action_space, **reward_net_kwargs
    )
    if agent_path is None:
        agent = stable_baselines3.PPO("MlpPolicy", venv, seed=_seed, **agent_kwargs)
    else:
        agent = stable_baselines3.PPO.load(agent_path, venv, seed=_seed, **agent_kwargs)
    if trajectory_path is None:
        # Setting the logger here is not really necessary (PreferenceComparisons
        # takes care of that automatically) but it avoids creating unnecessary loggers
        trajectory_generator = trainer.AgentTrainer(
            agent, reward_net, custom_logger=custom_logger
        )
    else:
        trajectory_generator = trainer.TrajectoryDataset(
            trajectory_path, _seed, custom_logger
        )

    fragmenter = preference_comparisons.RandomFragmenter(
        seed=_seed,
        custom_logger=custom_logger,
    )
    gatherer = preference_comparisons.SyntheticGatherer(
        **gatherer_kwargs, seed=_seed, custom_logger=custom_logger
    )
    reward_trainer = preference_comparisons.CrossEntropyRewardTrainer(
        model=reward_net, **reward_trainer_kwargs, custom_logger=custom_logger
    )
    main_trainer = preference_comparisons.PreferenceComparisons(
        trajectory_generator,
        reward_net,
        fragmenter=fragmenter,
        preference_gatherer=gatherer,
        reward_trainer=reward_trainer,
        comparisons_per_iteration=comparisons_per_iteration,
        fragment_length=fragment_length,
        transition_oversampling=transition_oversampling,
        custom_logger=custom_logger,
        allow_variable_horizon=allow_variable_horizon,
        preferences_path=preferences_path,
        seed=_seed,
    )
    results = main_trainer.train(total_timesteps, total_comparisons)

    th.save(reward_net, os.path.join(log_dir, "final_reward_net.pt"))

    if save_preferences:
        main_trainer.dataset.save(os.path.join(log_dir, "preferences.pkl"))

    # Storing and evaluating the policy only makes sense if we actually used it
    if trajectory_path is None and preferences_path is None:
        serialize.save_stable_model(
            os.path.join(log_dir, "final_policy"), agent, vec_normalize
        )
        sample_until = rollout.make_sample_until(
            min_timesteps=None, min_episodes=n_episodes_eval
        )
        trajs = rollout.generate_trajectories(
            agent,
            venv,
            sample_until=sample_until,
        )
        results["rollout"] = rollout.rollout_stats(trajs)

    return results


def main_console():
    observer = FileStorageObserver(
        os.path.join("output", "sacred", "train_preference_comparisons")
    )
    train_preference_comparisons_ex.observers.append(observer)
    train_preference_comparisons_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
