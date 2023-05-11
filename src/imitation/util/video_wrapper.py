"""Wrapper to record rendered video frames from an environment."""

import pathlib
from typing import Optional

import gym
from gym.wrappers.monitoring import video_recorder


def video_wrapper_factory(log_dir: pathlib.Path, **kwargs):
    """Returns a function that wraps the environment in a video recorder."""

    def f(env: gym.Env, i: int) -> VideoWrapper:
        """Wraps `env` in a recorder saving videos to `{log_dir}/videos/{i}`."""
        directory = log_dir / "videos" / str(i)
        return VideoWrapper(env, directory=directory, **kwargs)

    return f


class VideoWrapper(gym.Wrapper):
    """Creates videos from wrapped environment by calling render after each timestep."""

    episode_id: int
    video_recorder: Optional[video_recorder.VideoRecorder]
    single_video: bool
    directory: pathlib.Path

    def __init__(
        self,
        env: gym.Env,
        directory: pathlib.Path,
        single_video: bool = True,
        delete_on_close: bool = True,
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
            delete_on_close: if True, deletes the video file when the environment is
                closed. If False, the video file is left on disk.
        """
        super().__init__(env)
        self.episode_id = 0
        self.video_recorder = None
        self.single_video = single_video
        self.delete_on_close = delete_on_close
        self.current_video_path: Optional[pathlib.Path] = None

        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)

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
                base_path=str(self.directory / f"video.{self.episode_id:06}"),
                metadata={"episode_id": self.episode_id},
            )
            self.current_video_path = pathlib.Path(self.video_recorder.path)

    def reset(self, **kwargs):
        new_obs = super().reset(**kwargs)
        self._reset_video_recorder()
        self.episode_id += 1
        return new_obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self.video_recorder.capture_frame()
        # is it crazy to save the video path at every step?
        info["video_path"] = self.current_video_path
        return obs, rew, done, info

    def close(self) -> None:
        if self.video_recorder is not None:
            self.video_recorder.close()
            self.video_recorder = None
        if self.delete_on_close:
            for path in self.directory.glob("*.mp4"):
                path.unlink()
        super().close()
