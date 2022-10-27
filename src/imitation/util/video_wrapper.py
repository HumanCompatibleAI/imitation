"""Wrapper to record rendered video frames from an environment."""

import pathlib
from typing import Optional, Callable

import gym
from gym.wrappers.monitoring import video_recorder


class VideoWrapper(gym.Wrapper):
    """Creates videos from wrapped environment by calling render after each timestep."""

    episode_id: int
    video_recorder: Optional[video_recorder.VideoRecorder]
    single_video: bool
    directory: pathlib.Path
    cadence: int
    should_record: bool
    step_count: int

    def __init__(
        self,
        env: gym.Env,
        directory: pathlib.Path,
        single_video: bool = True,
        cadence: int = 1
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
            cadence: the video wrapper will save a video of the next episode that begins
                after every Nth step. So if cadence=100 and each episode has 30 steps, it will 
                record the 4th episode(first to start after step_count=100) and then the 7th
                episode (first to start after step_count=200).
        """
        super().__init__(env)
        self.episode_id = 0
        self.video_recorder = None
        self.single_video = single_video
        self.cadence = cadence

        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.should_record = False
        self.step_count = 0

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

        if self.video_recorder is None and (self.should_record or self.step_count % self.cadence == 0):
            # No video recorder -- start a new one.
            self.video_recorder = video_recorder.VideoRecorder(
                env=self.env,
                base_path=str(self.directory / f"video.{self.episode_id:06}"),
                metadata={"episode_id": self.episode_id},
            )
            self.should_record = False

    def reset(self):
        self._reset_video_recorder()
        self.episode_id += 1
        return self.env.reset()

    def step(self, action):
        res = self.env.step(action)
        self.step_count += 1
        if self.step_count % self.cadence == 0:
            self.should_record == 0
        if self.video_recorder != None:
            self.video_recorder.capture_frame()
        return res

    def close(self) -> None:
        if self.video_recorder is not None:
            self.video_recorder.close()
            self.video_recorder = None
        super().close()


def video_wrapper_factory(video_dir: pathlib.Path, cadence: int, **kwargs) -> Callable:
    def f(env: gym.Env, i: int)  -> VideoWrapper:
        """
        Returns a wrapper around a gym environment records a video if and only if i is 0

        Args:
            env: the environment to be wrapped around
            i: the index of the environment. This is to make the video wrapper compatible with
                    vectorized environments. Only environments with i=0 actually attach the VideoWrapper
        """

        return VideoWrapper(env, directory=video_dir, cadence=cadence, **kwargs) if i == 0 else env
    return f