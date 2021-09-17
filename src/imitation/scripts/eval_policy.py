"""Evaluate policies: render policy interactively, save videos, log episode return."""

import logging
import os
import os.path as osp
import time
from typing import Any, Mapping, Optional

import gym
from sacred.observers import FileStorageObserver
from stable_baselines3.common.vec_env import VecEnvWrapper

import imitation.util.sacred as sacred_util
from imitation.data import rollout, types
from imitation.policies import serialize
from imitation.rewards.serialize import load_reward
from imitation.scripts.config.eval_policy import eval_policy_ex
from imitation.util import reward_wrapper, util, video_wrapper


class InteractiveRender(VecEnvWrapper):
    """Render the wrapped environment(s) on screen."""

    def __init__(self, venv, fps):
        """Builds renderer for `venv` running at `fps` frames per second."""
        super().__init__(venv)
        self.render_fps = fps

    def reset(self):
        ob = self.venv.reset()
        self.venv.render()
        return ob

    def step_wait(self):
        ob = self.venv.step_wait()
        if self.render_fps > 0:
            time.sleep(1 / self.render_fps)
        self.venv.render()
        return ob


def video_wrapper_factory(log_dir: str, **kwargs):
    """Returns a function that wraps the environment in a video recorder."""

    def f(env: gym.Env, i: int) -> gym.Env:
        """Wraps `env` in a recorder saving videos to `{log_dir}/videos/{i}`."""
        directory = os.path.join(log_dir, "videos", str(i))
        return video_wrapper.VideoWrapper(env, directory=directory, **kwargs)

    return f


@eval_policy_ex.main
def eval_policy(
    _run,
    _seed: int,
    env_name: str,
    env_make_kwargs: Optional[Mapping[str, Any]],
    eval_n_timesteps: Optional[int],
    eval_n_episodes: Optional[int],
    num_vec: int,
    parallel: bool,
    render: bool,
    render_fps: int,
    videos: bool,
    video_kwargs: Mapping[str, Any],
    log_dir: str,
    policy_type: str,
    policy_path: str,
    reward_type: Optional[str] = None,
    reward_path: Optional[str] = None,
    max_episode_steps: Optional[int] = None,
    save_rollouts: bool = False,
):
    """Rolls a policy out in an environment, collecting statistics.

    Args:
        _seed: generated by Sacred.
        env_name: Gym environment identifier.
        env_make_kwargs: The kwargs passed to `spec.make` of a gym environment.
        eval_n_timesteps: Minimum number of timesteps to evaluate for. Set exactly
            one of `eval_n_episodes` and `eval_n_timesteps`.
        eval_n_episodes: Minimum number of episodes to evaluate for. Set exactly
            one of `eval_n_episodes` and `eval_n_timesteps`.
        num_vec: Number of environments to run simultaneously.
        parallel: If True, use `SubprocVecEnv` for true parallelism; otherwise,
            uses `DummyVecEnv`.
        max_episode_steps: If not None, then environments are wrapped by
            TimeLimit so that they have at most `max_episode_steps` steps per
            episode.
        render: If True, renders interactively to the screen.
        render_fps: The target number of frames per second to render on screen.
        videos: If True, saves videos to `log_dir`.
        video_kwargs: Keyword arguments passed through to `video_wrapper.VideoWrapper`.
        log_dir: The directory to log intermediate output to, such as episode reward.
        policy_type: A unique identifier for the saved policy,
            defined in POLICY_CLASSES.
        policy_path: A path to the serialized policy.
        reward_type: If specified, overrides the environment reward with
            a reward of this.
        reward_path: If reward_type is specified, the path to a serialized reward
            of `reward_type` to override the environment reward with.
        save_rollouts: whether to store the rollouts used for computing stats
            to disk (as pickle file).

    Returns:
        Return value of `imitation.util.rollout.rollout_stats()`.
    """
    os.makedirs(log_dir, exist_ok=True)
    sacred_util.build_sacred_symlink(log_dir, _run)

    logging.basicConfig(level=logging.INFO)
    logging.info("Logging to %s", log_dir)
    sample_until = rollout.make_sample_until(eval_n_timesteps, eval_n_episodes)
    post_wrappers = [video_wrapper_factory(log_dir, **video_kwargs)] if videos else None
    venv = util.make_vec_env(
        env_name,
        num_vec,
        seed=_seed,
        parallel=parallel,
        log_dir=log_dir,
        max_episode_steps=max_episode_steps,
        post_wrappers=post_wrappers,
        env_make_kwargs=env_make_kwargs,
    )

    try:
        if render:
            # As of July 31, 2020, DummyVecEnv rendering only works with num_vec=1
            # due to a bug on Stable Baselines 3.
            venv = InteractiveRender(venv, render_fps)

        if reward_type is not None:
            reward_fn = load_reward(reward_type, reward_path, venv)
            venv = reward_wrapper.RewardVecEnvWrapper(venv, reward_fn)
            logging.info(f"Wrapped env in reward {reward_type} from {reward_path}.")

        policy = serialize.load_policy(policy_type, policy_path, venv)
        trajs = rollout.generate_trajectories(policy, venv, sample_until)

        if save_rollouts:
            types.save(osp.join(log_dir, "rollouts.pkl"), trajs)

        return rollout.rollout_stats(trajs)
    finally:
        venv.close()


def main_console():
    observer = FileStorageObserver(osp.join("output", "sacred", "eval_policy"))
    eval_policy_ex.observers.append(observer)
    eval_policy_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
