"""Environment wrappers for collecting rollouts."""

import os
import shutil
import tempfile
import uuid
from typing import List, Optional, Sequence, Tuple

import cv2
import gym
import numpy as np
import numpy.typing as npt
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from imitation.data import rollout, types


class RenderImageInfoWrapper(gym.Wrapper):
    """Saves render images to `info`.

    Can be very memory intensive for large render images.
    Use `scale_factor` to reduce render image size.
    If you need to preserve the resolution and memory
    runs out, you can activate `use_file_cache` to save
    rendered images and instead put their path into `info`.
    """

    def __init__(
        self,
        env: gym.Env,
        scale_factor: float = 1.0,
        use_file_cache: bool = False,
    ):
        """Builds RenderImageInfoWrapper.

        Args:
            env: Environment to wrap.
            scale_factor: scales rendered images to be stored.
            use_file_cache: whether to save rendered images to disk.
        """
        super().__init__(env)
        self.scale_factor = scale_factor
        self.use_file_cache = use_file_cache
        if self.use_file_cache:
            self.file_cache = tempfile.mkdtemp("imitation_RenderImageInfoWrapper")

    def step(self, action):
        obs, rew, done, info = self.env.step(action)

        if self._active:
            rendered_image = self.render(mode="rgb_array")
            # Scale the render image
            scaled_size = (
                int(self.scale_factor * rendered_image.shape[0]),
                int(self.scale_factor * rendered_image.shape[1]),
            )
            scaled_rendered_image = cv2.resize(
                rendered_image,
                scaled_size,
                interpolation=cv2.INTER_AREA,
            )
            # Store the render image
            if not self.use_file_cache:
                info["rendered_img"] = scaled_rendered_image
            else:
                unique_file_path = os.path.join(
                    self.file_cache,
                    str(uuid.uuid4()) + ".npy",
                )
                np.save(unique_file_path, scaled_rendered_image)
                info["rendered_img"] = unique_file_path

        # Do not show window of classic control envs
        if self.env.viewer is not None and self.env.viewer.window.visible:
            self.env.viewer.window.set_visible(False)

        return obs, rew, done, info

    def close(self) -> None:
        if self.use_file_cache:
            shutil.rmtree(self.file_cache)
        return super().close()


class BufferingWrapper(VecEnvWrapper):
    """Saves transitions of underlying VecEnv.

    Retrieve saved transitions using `pop_transitions()`.
    """

    error_on_premature_event: bool
    _trajectories: List[types.TrajectoryWithRew]
    _ep_lens: List[int]
    _init_reset: bool
    _traj_accum: Optional[rollout.TrajectoryAccumulator]
    _timesteps: Optional[npt.NDArray[np.int_]]
    n_transitions: Optional[int]

    def __init__(self, venv: VecEnv, error_on_premature_reset: bool = True):
        """Builds BufferingWrapper.

        Args:
            venv: The wrapped VecEnv.
            error_on_premature_reset: Error if `reset()` is called on this wrapper
                and there are saved samples that haven't yet been accessed.
        """
        super().__init__(venv)
        self.error_on_premature_reset = error_on_premature_reset
        self._trajectories = []
        self._ep_lens = []
        self._init_reset = False
        self._traj_accum = None
        self._saved_acts = None
        self._timesteps = None
        self.n_transitions = None

    def reset(self, **kwargs):
        if (
            self._init_reset
            and self.error_on_premature_reset
            and self.n_transitions > 0
        ):  # noqa: E127
            raise RuntimeError("BufferingWrapper reset() before samples were accessed")
        self._init_reset = True
        self.n_transitions = 0
        obs = self.venv.reset(**kwargs)
        self._traj_accum = rollout.TrajectoryAccumulator()
        for i, ob in enumerate(obs):
            self._traj_accum.add_step({"obs": ob}, key=i)
        self._timesteps = np.zeros((len(obs),), dtype=int)
        return obs

    def step_async(self, actions):
        assert self._init_reset
        assert self._saved_acts is None
        self.venv.step_async(actions)
        self._saved_acts = actions

    def step_wait(self):
        assert self._init_reset
        assert self._saved_acts is not None
        acts, self._saved_acts = self._saved_acts, None
        obs, rews, dones, infos = self.venv.step_wait()

        self.n_transitions += self.num_envs
        self._timesteps += 1
        ep_lens = self._timesteps[dones]
        if len(ep_lens) > 0:
            self._ep_lens += list(ep_lens)
        self._timesteps[dones] = 0

        finished_trajs = self._traj_accum.add_steps_and_auto_finish(
            acts,
            obs,
            rews,
            dones,
            infos,
        )
        self._trajectories.extend(finished_trajs)

        return obs, rews, dones, infos

    def _finish_partial_trajectories(self) -> Sequence[types.TrajectoryWithRew]:
        """Finishes and returns partial trajectories in `self._traj_accum`."""
        assert self._traj_accum is not None
        trajs = []
        for i in range(self.num_envs):
            # Check that we have any transitions at all.
            # The number of "transitions" or "timesteps" stored for the ith
            # environment is the number of step dicts stored in
            # `partial_trajectories[i]` minus one. We need to offset by one because
            # the first step dict is comes from `reset()`, not from `step()`.
            n_transitions = len(self._traj_accum.partial_trajectories[i]) - 1
            assert n_transitions >= 0, "Invalid TrajectoryAccumulator state"
            if n_transitions >= 1:
                traj = self._traj_accum.finish_trajectory(i, terminal=False)
                trajs.append(traj)

                # Reinitialize a partial trajectory starting with the final observation.
                self._traj_accum.add_step({"obs": traj.obs[-1]}, key=i)
        return trajs

    def pop_finished_trajectories(
        self,
    ) -> Tuple[Sequence[types.TrajectoryWithRew], Sequence[int]]:
        """Pops recorded complete trajectories `trajs` and episode lengths `ep_lens`.

        Returns:
            A tuple `(trajs, ep_lens)` where `trajs` is a sequence of trajectories
            including the terminal state (but possibly missing initial states, if
            `pop_trajectories` was previously called) and `ep_lens` is a sequence
            of episode lengths. Note the episode length will be longer than the
            trajectory length when the trajectory misses initial states.
        """
        trajectories = self._trajectories
        ep_lens = self._ep_lens
        self._trajectories = []
        self._ep_lens = []
        self.n_transitions = 0
        return trajectories, ep_lens

    def pop_trajectories(
        self,
    ) -> Tuple[Sequence[types.TrajectoryWithRew], Sequence[int]]:
        """Pops recorded trajectories `trajs` and episode lengths `ep_lens`.

        Returns:
            A tuple `(trajs, ep_lens)`. `trajs` is a sequence of trajectory fragments,
            consisting of data collected after the last call to `pop_trajectories`.
            They may miss initial states (if `pop_trajectories` previously returned
            a fragment for that episode), and terminal states (if the episode has
            yet to complete). `ep_lens` is the total length of completed episodes.
        """
        if self.n_transitions == 0:
            return [], []
        partial_trajs = self._finish_partial_trajectories()
        self._trajectories.extend(partial_trajs)
        return self.pop_finished_trajectories()

    def pop_transitions(self) -> types.TransitionsWithRew:
        """Pops recorded transitions, returning them as an instance of Transitions.

        Returns:
            All transitions recorded since the last call.

        Raises:
            RuntimeError: empty (no transitions recorded since last pop).
        """
        if self.n_transitions == 0:
            # It would be better to return an empty `Transitions`, but we would need
            # to get the non-zero dimensions of every np.ndarray attribute correct to
            # avoid downstream errors. This is easier and sufficient for now.
            raise RuntimeError("Called pop_transitions on an empty BufferingWrapper")
        # make a copy for the assert later
        n_transitions = self.n_transitions
        trajectories, _ = self.pop_trajectories()
        transitions = rollout.flatten_trajectories_with_rew(trajectories)
        assert len(transitions.obs) == n_transitions
        return transitions


class RolloutInfoWrapper(gym.Wrapper):
    """Add the entire episode's rewards and observations to `info` at episode end.

    Whenever done=True, `info["rollouts"]` is a dict with keys "obs" and "rews", whose
    corresponding values hold the NumPy arrays containing the raw observations and
    rewards seen during this episode.
    """

    def __init__(self, env: gym.Env):
        """Builds RolloutInfoWrapper.

        Args:
            env: Environment to wrap.
        """
        super().__init__(env)
        self._obs = None
        self._rews = None

    def reset(self, **kwargs):
        new_obs = super().reset(**kwargs)
        self._obs = [new_obs]
        self._rews = []
        return new_obs

    def step(self, action):
        obs, rew, done, info = self.env.step(action)
        self._obs.append(obs)
        self._rews.append(rew)

        if done:
            assert "rollout" not in info
            info["rollout"] = {
                "obs": np.stack(self._obs),
                "rews": np.stack(self._rews),
            }
        return obs, rew, done, info
