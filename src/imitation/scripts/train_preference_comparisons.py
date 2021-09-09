"""Train a reward model using preference comparisons.

Can be used as a CLI script, or the `train_and_plot` function can be called directly.
"""

import os
from typing import Any, Dict, Mapping, Optional

import stable_baselines3
import torch as th
from sacred.observers import FileStorageObserver

from imitation.algorithms import preference_comparisons
from imitation.data import rollout
from imitation.policies import trainer
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
    env_make_kwargs: Optional[Mapping[str, Any]],
    num_vec: int,
    parallel: bool,
    max_episode_steps: Optional[int],
    log_dir: str,
    iterations: int,
    sample_steps: int,
    agent_steps: int,
    fragment_length: int,
    num_pairs: int,
    n_episodes_eval: int,
    reward_kwargs: Dict[str, Any],
    agent_kwargs: Dict[str, Any],
    allow_variable_horizon: bool,
) -> dict:
    """Train a reward model using preference comparisons.

    Args:
        _seed: Random seed.
        env_name: The environment to train in.
        env_make_kwargs: The kwargs passed to `spec.make` of a gym environment.
        num_vec: Number of `gym.Env` to vectorize.
        parallel: Whether to use "true" parallelism. If True, then use `SubProcVecEnv`.
            Otherwise, use `DummyVecEnv` which steps through environments serially.
        max_episode_steps: If not None, then a TimeLimit wrapper is applied to each
            environment to artificially limit the maximum number of timesteps in an
            episode.
        log_dir: Directory to save models and other logging to.
        iterations: Number of iterations of the outer training loop
            (i.e. number of times that new preferences are collected).
        sample_steps: how many environment steps to sample each time that
            preferences are collected. Trajectory fragments will be chosen
            from all the sampled steps.
        agent_steps: how many environment steps to train the agent for in
            each iteration.
        fragment_length: number of transitions in each fragment that is shown
            for preference comparisons.
        num_pairs: number of fragment pairs to collect preferences for in
            each iteration.
        n_episodes_eval: The number of episodes to average over when calculating
            the average episode reward of the learned policy for return.
        reward_kwargs: passed to BasicRewardNet
        agent_kwargs: passed to SB3's PPO
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
        env_make_kwargs=env_make_kwargs,
    )

    reward_net = reward_nets.BasicRewardNet(
        venv.observation_space, venv.action_space, **reward_kwargs
    )
    agent = stable_baselines3.PPO("MlpPolicy", venv, **agent_kwargs)
    agent_trainer = trainer.AgentTrainer(agent, reward_net)
    fragmenter = preference_comparisons.RandomFragmenter(
        fragment_length=fragment_length, num_pairs=num_pairs, seed=_seed
    )
    main_trainer = preference_comparisons.PreferenceComparisons(
        agent_trainer,
        reward_net,
        sample_steps=sample_steps,
        agent_steps=agent_steps,
        fragmenter=fragmenter,
        custom_logger=custom_logger,
        allow_variable_horizon=allow_variable_horizon,
    )
    main_trainer.train(iterations)

    agent.save(os.path.join(log_dir, "final_agent"))
    th.save(reward_net.state_dict(), os.path.join(log_dir, "final_reward_net.pt"))

    sample_until = rollout.make_sample_until(
        min_timesteps=None, min_episodes=n_episodes_eval
    )
    trajs = rollout.generate_trajectories(
        agent,
        venv,
        sample_until=sample_until,
    )
    return rollout.rollout_stats(trajs)


def main_console():
    observer = FileStorageObserver(
        os.path.join("output", "sacred", "train_preference_comparisons")
    )
    train_preference_comparisons_ex.observers.append(observer)
    train_preference_comparisons_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
