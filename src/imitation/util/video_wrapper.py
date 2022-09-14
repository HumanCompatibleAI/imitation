"""Wrapper to record rendered video frames from an environment."""

import os
import os.path as osp
from typing import Any, Mapping, Optional

import gym
import stable_baselines3.common.logger as sb_logger
from gym.wrappers.monitoring import video_recorder
from stable_baselines3.common import callbacks, policies, vec_env

from imitation.data import rollout, types


class VideoWrapper(gym.Wrapper):
    """Creates videos from wrapped environment by calling render after each timestep."""

    def __init__(
        self,
        env: gym.Env,
        directory: types.AnyPath,
        single_video: bool = True,
    ):
        """Builds a VideoWrapper.

        Args:
            env: the wrapped environment.
            directory: the output directory.
            single_video: if True, generates a single video file, with episodes
                concatenated. If False, a new video file is created for each episode.
                Usually a single video file is what is desired. However, if one is
                searching for an interesting episode (perhaps by looking at the
                metadata), then saving to different files can be useful.
        """
        super().__init__(env)
        self._episode_id = 0
        self.video_recorder = None
        self.single_video = single_video

        self.directory = os.path.abspath(directory)
        os.makedirs(self.directory, exist_ok=True)

    @property
    def episode_id(self) -> int:
        return self._episode_id

    def _reset_video_recorder(self) -> None:
        """Creates a video recorder if one does not already exist.

        Called at the start of each episode (by `reset`). When a video recorder is
        already present, it will only create a new one if `self.single_video == False`.
        """
        if self.video_recorder is not None:
            # Video recorder already started.
            if not self.single_video:
                # We want a new video for each episode, so destroy current recorder.
                self.video_recorder.close()
                self.video_recorder = None

        if self.video_recorder is None:
            # No video recorder -- start a new one.
            self.video_recorder = video_recorder.VideoRecorder(
                env=self.env,
                base_path=os.path.join(
                    self.directory,
                    "video.{:06}".format(self._episode_id),
                ),
                metadata={"episode_id": self._episode_id},
            )

    def reset(self):
        self._reset_video_recorder()
        self._episode_id += 1
        return self.env.reset()

    def step(self, action):
        res = self.env.step(action)
        self.video_recorder.capture_frame()
        return res

    def close(self) -> None:
        if self.video_recorder is not None:
            self.video_recorder.close()
            self.video_recorder = None
        super().close()


def record_and_save_video(
    output_dir: str,
    policy: policies.BasePolicy,
    eval_venv: vec_env.VecEnv,
    video_kwargs: Optional[Mapping[str, Any]] = None,
    logger: Optional[sb_logger.Logger] = None,
) -> None:
    video_dir = osp.join(output_dir, "videos")
    video_venv = VideoWrapper(
        eval_venv,
        directory=video_dir,
        **(video_kwargs or dict()),
    )
    sample_until = rollout.make_sample_until(min_timesteps=None, min_episodes=1)
    # video.{:06}.mp4".format(VideoWrapper.episode_id) will be saved within
    # rollout.generate_trajectories()
    rollout.generate_trajectories(policy, video_venv, sample_until)
    video_name = "video.{:06}.mp4".format(video_venv.episode_id - 1)
    assert video_name in os.listdir(video_dir)
    video_path = osp.join(video_dir, video_name)
    if logger:
        logger.record("video", video_path)
        logger.log(f"Recording and saving video to {video_path} ...")


class SaveVideoCallback(callbacks.EventCallback):
    """Saves the policy using `record_and_save_video` each time when it is called.

    Should be used in conjunction with `callbacks.EveryNTimesteps`
    or another event-based trigger.
    """

    def __init__(
        self,
        policy_dir: str,
        eval_venv: vec_env.VecEnv,
        *args,
        video_kwargs: Optional[Mapping[str, Any]] = None,
        **kwargs,
    ):
        """Builds SavePolicyCallback.

        Args:
            policy_dir: Directory to save checkpoints.
            eval_venv: Environment to evaluate the policy on.
            *args: Passed through to `callbacks.EventCallback`.
            video_kwargs: Keyword arguments to pass to `VideoWrapper`.
            **kwargs: Passed through to `callbacks.EventCallback`.
        """
        super().__init__(*args, **kwargs)
        self.policy_dir = policy_dir
        self.eval_venv = eval_venv
        self.video_kwargs = video_kwargs or dict()

    def _on_step(self) -> bool:
        output_dir = os.path.join(self.policy_dir, f"{self.num_timesteps:012d}")
        record_and_save_video(
            output_dir=output_dir,
            policy=self.model.policy,
            eval_venv=self.eval_venv,
            video_kwargs=self.video_kwargs,
            logger=self.model.logger,
        )

        return True
