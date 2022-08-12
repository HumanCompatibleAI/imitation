import logging
import math
from multiprocessing import cpu_count, Pipe, Process
from multiprocessing.connection import Connection
from typing import Callable, Tuple, Optional

import abc
import itertools

import gym
import numpy as np
import torch

from imitation.rewards.reward_distance.transition_sampler import TransitionSampler, ActionSampler

from gym.envs.mujoco.half_cheetah_v3 import HalfCheetahEnv
from gym.envs.mujoco.ant_v3 import AntEnv
from gym.wrappers.time_limit import TimeLimit


class GetObsWrapper(gym.Wrapper):
    """Gym wrappers (such as TimeLimit wrapper) do not implement a _get_obs() method.
       This wrapper simply adds that functionality.
    """
    def step(self, action):
        return super().step(action)

    def _get_obs(self):
        return self.unwrapped._get_obs()



def _get_obs_to_qpos_qvel_converter_for_env(env: gym.Env) -> Callable:
    """Gets a callable that converts an obs into the state necessary to reset the simulator.

    TODO(redacted): Pass an object to the transition sampler that handles this instead of
    having to define all these in this file!

    Args:
        env: The env for which to retrieve the converter.

    Returns:
        The converter for the env.
    """
    def _half_cheetah_obs_to_qpos_qvel_converter(obs: np.ndarray) -> np.ndarray:
        return obs[..., :9], obs[..., 9:]

    def _ant_obs_to_qpos_qvel_converter(obs: np.ndarray) -> np.ndarray:
        return obs[..., :15], obs[..., 15:15+14]

    # TODO (usman): Is there a general way to check if the gym environment is wrapped?
    if isinstance(env, TimeLimit) or isinstance(env, GetObsWrapper):
        env_to_check = env.unwrapped
    else:
        env_to_check = env
    if isinstance(env_to_check, HalfCheetahEnv):
        assert env_to_check._exclude_current_positions_from_observation is False, (
                "For proper dynamics solution, please include current position in obs")
        return _half_cheetah_obs_to_qpos_qvel_converter
    elif isinstance(env_to_check, AntEnv):
        assert env_to_check._exclude_current_positions_from_observation is False, (
                "For proper dynamics solution, please include current position in obs")
        return _ant_obs_to_qpos_qvel_converter
    #elif isinstance(env, CustomReacherEnv):
    #    return _custom_reacher_env_obs_to_qpos_qvel_converter
    else:
        raise ValueError(f"No converter for env {type(env)} implemented.")


class CloudpickleWrapper(object):
    """Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle).
••••
    Args:
        x: The content to pickle.
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def _worker(remote: Connection, make_env_fn: CloudpickleWrapper) -> None:
    """Worker for propagating Mujoco Dynamics.•

    Should be spawned by `Process` and uses `Pipe` connections to communicate with main thread.

    Args:
        remote: handle for commuicating with parent thread.
        make_env_fn: Function to spawn the env, wrapped in a `CloudpickleWrapper` to avoid serialization issues.
    """
    env = GetObsWrapper(make_env_fn.x())
    env.reset()
    obs_to_sim_state_converter = _get_obs_to_qpos_qvel_converter_for_env(env)

    try:
        while True:
            cmd, data = remote.recv()
            if cmd == "propagate":
                states, actions = data

                # If using one state per worker, unsqueeze for following for loop.
                if len(states.shape) < 2:
                    states = np.expand_dims(states, 0)
                    actions = np.expand_dims(actions, 0)

                num_states, state_dim = states.shape
                num_actions = actions.shape[1]
                next_states = np.empty((num_states, num_actions, state_dim), dtype=states.dtype)

                qpos, qvel = obs_to_sim_state_converter(states)
                for state_index in range(num_states):
                    for action_index in range(num_actions):
                        env.set_state(qpos[state_index], qvel[state_index])
                        env.do_simulation(actions[state_index, action_index], env.frame_skip)
                        next_states[state_index, action_index] = env._get_obs()

                remote.send(next_states)
            elif cmd == "close":
                break
            else:
                raise NotImplementedError(f"{cmd} is not implemented in the worker")

    except KeyboardInterrupt:
        print("Mujoco Sampler Worker: got KeyboardInterrupt")

    finally:
        remote.close()
        env.close()


class MujocoTransitionSampler(TransitionSampler):
    """Samples transitions for mujoco environments using sampled actions and the ground truth simulator.

    Args:
        make_env_fn: Function that creates a mujoco env when called.
        action_sampler: A callable object that samples actions.
        num_workers: Number of worker processes to spawn; 0 will use all cpu cores
        worker_timeout: (s) time to wait for worker response before failing
    """
    def __init__(
            self,
            make_env_fn: Callable,
            action_sampler: ActionSampler,
            num_workers: int = 0,
            worker_timeout: float = 10,
    ):
        self.worker_timeout = worker_timeout
        self.num_workers = num_workers if num_workers > 0 else cpu_count()
        self.remotes, self.worker_remotes, self.worker_processes = [[], [], []]
        self.closed = True
        self.spawn_workers(make_env_fn)

        self.action_sampler = action_sampler

        # Alias for readability in the implementation.
        self.num_actions = self.action_sampler.num_actions

    def spawn_workers(self, make_env_fn: Callable) -> None:
        """Creates remote workers that will propagate Mujoco dynamics.

        Args:
            make_env_fn: The function that creates the environment.
        """
        # Pipes allow communication between this class and the workers.
        self.remotes, self.worker_remotes = zip(*[Pipe() for _ in range(self.num_workers)])

        self.worker_processes = [
            Process(target=_worker, args=(remote, CloudpickleWrapper(make_env_fn))) for remote in self.worker_remotes
        ]

        for ps in self.worker_processes:
            # If main process crosses, we should not cause things to hang
            ps.daemon = True
            ps.start()

        self.closed = False

    def assert_not_closed(self) -> None:
        """Asserts that the processes for stepping dynamics are not closed."""
        assert not self.closed, ("Attempted to execute command without spawned worker processes. "
                                 "Please ensure `spawn_workers` has been called first")

    def close(self) -> None:
        """Closes the processes used to step dynamics."""
        if self.closed:
            return

        for remote in self.remotes:
            remote.send(("close", None))
            remote.close()

        for ps in self.worker_processes:
            ps.join()

        self.closed = True

    def __del__(self):
        self.close()

    def sample(self, states: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """See base class documentation."""
        assert states.ndim == 2
        num_states = len(states)
        actions = self.action_sampler(num_states, states.dtype, states.device)
        next_states = self._step_dynamics(states, actions)
        weights = torch.ones((num_states, self.num_actions), dtype=states.dtype, device=states.device)
        return actions, next_states, weights

    @property
    def num_transitions_per_state(self) -> int:
        return self.action_sampler.num_actions

    def _step_dynamics(self, states: torch.Tensor, actions: torch.Tensor) -> torch.Tensor:
        """Steps the provided states forward applying actions.

        Args:
            states: The states to step forward. Shape of (num_states, state_dim).
            actions: The actions to apply. Shape of (num_states, num_actions, action_dim).

        Returns:
            The next states of shape (num_states, num_actions, state_dim).
        """
        self.assert_not_closed()

        # Do all the operations with numpy arrays.
        tensor_device = states.device

        states = states.detach().cpu().numpy()
        actions = actions.detach().cpu().numpy()
        num_states, state_dim = states.shape

        if num_states < self.num_workers:
            logging.warning(("State size ({}) less than num_workers ({}). "
                          "Ideally, num_workers <= state_size".format(num_states, self.num_workers)))

        next_states = np.empty((num_states, self.num_actions, state_dim), dtype=states.dtype)
        step_size = math.ceil(num_states / self.num_workers)
        for remote_index, state_index in enumerate(range(0, num_states, step_size)):
            upper_index = min(num_states, state_index + step_size)
            worker_data = states[state_index:upper_index], actions[state_index:upper_index]
            self.remotes[remote_index].send(("propagate", worker_data))

        for remote_index, state_index in enumerate(range(0, num_states, step_size)):
            assert self.remotes[remote_index].poll(
                self.worker_timeout), f"Timed out waiting for response from worker: {remote_index}/{len(self.remotes)}"
            upper_index = min(num_states, state_index + step_size)
            next_states[state_index:upper_index] = self.remotes[remote_index].recv()

        return torch.tensor(next_states, device=tensor_device)




if __name__ == '__main__':
    from imitation.rewards.reward_distance.transition_sampler import UniformlyRandomActionSampler
    from imitation.rewards.reward_distance.mujoco_sampler import MujocoTransitionSampler
    unif_actions = UniformlyRandomActionSampler(20, 1, 6)
    transition_sampler = MujocoTransitionSampler(
            make_env_fn = lambda :gym.make('HalfCheetah-v3', exclude_current_positions_from_observation=False),
            action_sampler=unif_actions,
            num_workers=2,
            worker_timeout=10)
    env = gym.make('HalfCheetah-v3', exclude_current_positions_from_observation=False)
    transition_sampler.sample(torch.from_numpy(np.concatenate([env.observation_space.sample()[None,...] for _ in range(10)], axis=0)))