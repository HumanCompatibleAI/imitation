"""Environment wrappers for collecting rollouts."""

from typing import Dict, List, Optional, Sequence, Tuple, Union

import gymnasium as gym
import numpy as np
import numpy.typing as npt
from gymnasium.core import Env
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from imitation.data import rollout, types

# The key for human readable data in the observation.
HR_OBS_KEY = "HR_OBS"


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
        obs = types.maybe_wrap_in_dictobs(obs)
        for i, ob in enumerate(obs):
            self._traj_accum.add_step({"obs": ob}, key=i)
        self._timesteps = np.zeros((len(obs),), dtype=int)
        obs = types.maybe_unwrap_dictobs(obs)
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
    """Adds the entire episode's rewards and observations to `info` at episode end.

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
        new_obs, info = super().reset(**kwargs)
        self._obs = [types.maybe_wrap_in_dictobs(new_obs)]
        self._rews = []
        return new_obs, info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        done = terminated or truncated
        self._obs.append(types.maybe_wrap_in_dictobs(obs))
        self._rews.append(rew)

        if done:
            assert "rollout" not in info
            info["rollout"] = {
                "obs": types.stack_maybe_dictobs(self._obs),
                "rews": np.stack(self._rews),
            }
        return obs, rew, terminated, truncated, info


class HumanReadableWrapper(gym.Wrapper):
    """Adds human-readable observation to `obs` at every step."""

    def __init__(self, env: Env, original_obs_key: str = "ORI_OBS"):
        """Builds HumanReadableWrapper.

        Args:
            env: Environment to wrap.
            original_obs_key: The key for original observation if the original
                observation is not in dict format.

        Raises:
            ValueError: If `env.render_mode` is not "rgb_array".

        """
        if env.render_mode != "rgb_array":
            raise ValueError(
                "HumanReadableWrapper requires render_mode='rgb_array', "
                f"got {env.render_mode!r}",
            )
        self._original_obs_key = original_obs_key
        super().__init__(env)
        self._update_obs_space()

    def _update_obs_space(self):
        # need to reset before render.
        self.env.reset()
        example_rgb_obs = self.env.render()
        new_rgb_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=example_rgb_obs.shape,
            dtype=np.uint8,
        )
        curr_sapce = self.observation_space
        if isinstance(curr_sapce, gym.spaces.Dict):
            curr_sapce.spaces[HR_OBS_KEY] = new_rgb_space
        else:
            self.observation_space = gym.spaces.Dict(
                {
                    HR_OBS_KEY: new_rgb_space,
                    self._original_obs_key: curr_sapce,
                },
            )

    def _add_hr_obs(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> Dict[str, np.ndarray]:
        """Adds human-readable observation to obs.

        Transforms obs into dictionary if it is not already, and adds the human-readable
        observation from `env.render()` under the key HR_OBS_KEY.

        Args:
            obs: Observation from environment.

        Returns:
            Observation dictionary with the human-readable data.

        Raises:
            KeyError: When the key HR_OBS_KEY already exists in the observation
                dictionary.
        """
        if not isinstance(obs, Dict):
            obs = {self._original_obs_key: obs}

        if HR_OBS_KEY in obs:
            raise KeyError(f"{HR_OBS_KEY!r} already exists in observation dict")
        obs[HR_OBS_KEY] = self.env.render()  # type: ignore[assignment]
        return obs

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self._add_hr_obs(obs), info

    def step(self, action):
        obs, rew, terminated, truncated, info = self.env.step(action)
        return self._add_hr_obs(obs), rew, terminated, truncated, info


def remove_rgb_obs_space(obs_space: gym.Space) -> gym.Space:
    """Removes rgb observation space from the observation space."""
    if not isinstance(obs_space, gym.spaces.Dict):
        return obs_space
    if HR_OBS_KEY not in obs_space.spaces:
        return obs_space
    if len(obs_space.keys()) == 1:
        raise ValueError(
            "Only human readable observation space exists, can't remove it",
        )
    # keeps the original obs_space unchanged in case it is used elsewhere.
    new_obs_space = gym.spaces.Dict(obs_space.spaces.copy())
    del new_obs_space.spaces[HR_OBS_KEY]
    if len(new_obs_space.spaces) == 1:
        # unwrap dictionary structure
        return next(iter(new_obs_space.values()))
    return new_obs_space
