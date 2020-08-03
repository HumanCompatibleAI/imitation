from typing import List

import gym
import numpy as np
from stable_baselines3.common.vec_env import VecEnv, VecEnvWrapper

from imitation.data import rollout, types


class BufferingWrapper(VecEnvWrapper):
    """Saves transitions of underlying VecEnv.

    Retrieve saved transitions using `pop_transitions()`.
    """

    def __init__(self, venv: VecEnv, error_on_premature_reset: bool = True):
        """
        Args:
          venv: The wrapped VecEnv.
          error_on_premature_reset: Error if `reset()` is called on this wrapper
            and there are saved samples that haven't yet been accessed.
        """
        super().__init__(venv)
        self.error_on_premature_reset = error_on_premature_reset
        self._trajectories = []
        self._init_reset = False
        self._traj_accum = None
        self._saved_acts = None
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
        finished_trajs = self._traj_accum.add_steps_and_auto_finish(
            acts, obs, rews, dones, infos
        )
        self._trajectories.extend(finished_trajs)
        self.n_transitions += self.num_envs
        return obs, rews, dones, infos

    def _finish_partial_trajectories(self) -> List[types.TrajectoryWithRew]:
        """Finishes and returns partial trajectories in `self._traj_accum`."""
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
                traj = self._traj_accum.finish_trajectory(i)
                trajs.append(traj)

                # Reinitialize a partial trajectory starting with the final observation.
                self._traj_accum.add_step({"obs": traj.obs[-1]}, key=i)
        return trajs

    def pop_transitions(self) -> types.TransitionsWithRew:
        """Pops recorded transitions, returning them as an instance of Transitions.

        Raises a RuntimeError if called when `self.n_transitions == 0`.
        """
        if self.n_transitions == 0:
            # It would be better to return an empty `Transitions`, but we would need
            # to get the non-zero dimensions of every np.ndarray attribute correct to
            # avoid downstream errors. This is easier and sufficient for now.
            raise RuntimeError("Called pop_transitions on an empty BufferingWrapper")
        partial_trajs = self._finish_partial_trajectories()
        self._trajectories.extend(partial_trajs)
        transitions = rollout.flatten_trajectories_with_rew(self._trajectories)
        assert len(transitions.obs) == self.n_transitions
        self._trajectories = []
        self.n_transitions = 0
        return transitions


class RolloutInfoWrapper(gym.Wrapper):
    """Add the entire episode's rewards and observations to `info` at episode end.

    Whenever done=True, `info["rollouts"]` is a dict with keys "obs" and "rews", whose
    corresponding values hold the Numpy arrays containing the raw observations and
    rewards seen during this episode.
    """

    def __init__(self, env):
        super().__init__(env)
        self._obs = None
        self._rews = None

    def reset(self, **kwargs):
        new_obs = super().reset()
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
