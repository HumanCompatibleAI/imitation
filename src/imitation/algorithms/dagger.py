"""DAgger (https://arxiv.org/pdf/1011.0686.pdf).

Interactively trains policy by collecting some demonstrations, doing BC, collecting more
demonstrations, doing BC again, etc. Initially the demonstrations just come from the
expert's policy; over time, they shift to be drawn more and more from the imitator's
policy.
"""

import abc
import dataclasses
import logging
import os
from typing import Callable, Tuple, Union

import gym
import numpy as np
import torch as th
from stable_baselines3.common import utils
from torch.utils import data as th_data

from imitation.algorithms import bc
from imitation.data import rollout, types
from imitation.util import util


class BetaSchedule(abc.ABC):
    """
    Determines the value of beta (% of time that demonstrator action is used) over the
    progression of training rounds.
    """

    @abc.abstractmethod
    def __call__(self, round_num: int) -> float:
        """Gives the value of beta for the current round.

        Args:
            round: the current round number. Rounds are assumed to be numbered 0, 1, 2,
              etc.

        Returns:
            beta: the fraction of the time to sample a demonstrator action. Robot
              actions will be sampled the remainder of the time.
        """


class LinearBetaSchedule(BetaSchedule):
    """
    Linearly-decreasing schedule for beta (% of time that demonstrator action is used).
    """

    def __init__(self, rampdown_rounds: int):
        """
        Args:
            rampdown_rounds: number of rounds over which to anneal beta.
        """
        self.rampdown_rounds = rampdown_rounds

    def __call__(self, round_num: int) -> float:
        assert round_num >= 0
        return min(1, max(0, (self.rampdown_rounds - round_num) / self.rampdown_rounds))


def reconstruct_trainer(
    scratch_dir: str, device: Union[th.device, str] = "auto"
) -> "DAggerTrainer":
    """Reconstruct trainer from the latest snapshot in some working directory.

    Args:
      scratch_dir: path to the working directory created by a previous run of
        this algorithm. The directory should contain `checkpoint-latest.pt` and
        `policy-latest.pt` files.
      device: device on which to load the trainer.

    Returns:
      trainer: a reconstructed `DAggerTrainer` with the same state as the
        previously-saved one.
    """
    checkpoint_path = os.path.join(scratch_dir, "checkpoint-latest.pt")
    return th.load(checkpoint_path, map_location=utils.get_device(device))


def _save_trajectory(
    npz_path: str,
    trajectory: types.Trajectory,
) -> None:
    """Save a trajectory as a compressed Numpy file."""
    save_dir = os.path.dirname(npz_path)
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
    assert isinstance(trajectory, types.Trajectory)
    np.savez_compressed(npz_path, **dataclasses.asdict(trajectory))


def _load_trajectory(npz_path: str) -> types.Trajectory:
    """Load a single trajectory from a compressed Numpy file."""
    np_data = np.load(npz_path, allow_pickle=True)
    has_rew = "rews" in np_data
    cls = types.TrajectoryWithRew if has_rew else types.Trajectory
    return cls(**dict(np_data.items()))


class InteractiveTrajectoryCollector(gym.Wrapper):
    """Wrapper around the `.step()` and `.reset()` of an env that allows DAgger to
    inject a "robot" action (i.e. an action from the imitation policy) that overrides
    the action given to `.step()` when necessary.

    Will also automatically save trajectories.
    """

    def __init__(
        self,
        env: gym.Env,
        get_robot_act: Callable[[np.ndarray], np.ndarray],
        beta: float,
        save_dir: str,
    ):
        """Trajectory collector constructor.

        Args:
          env: environment to sample trajectories from.
          get_robot_act: get a single robot action that can be substituted for
              human action. Takes a single observation as input & returns a
              single action.
          beta: fraction of the time to use action given to .step() instead of
              robot action.
          save_dir: directory to save collected trajectories in.
        """
        super().__init__(env)
        self.get_robot_act = get_robot_act
        assert 0 <= beta <= 1
        self.beta = beta
        self.traj_accum = None
        self.save_dir = save_dir
        self._last_obs = None
        self._done_before = True
        self._is_reset = False

    def reset(self) -> np.ndarray:
        """Resets the environment.

        Returns:
            obs: first observation of a new trajectory.
        """
        self.traj_accum = rollout.TrajectoryAccumulator()
        obs = self.env.reset()
        self._last_obs = obs
        self.traj_accum.add_step({"obs": obs})
        self._done_before = False
        self._is_reset = True
        return obs

    def step(self, user_action: np.ndarray) -> Tuple[np.ndarray, float, bool, dict]:
        """Steps the environment.

        DAgger needs to be able to inject imitation policy actions randomly at some
        subset of time steps. This method will replace the given action with a
        "robot" (i.e. imitation policy) action if necessary.

        Args:
          user_action: the _intended_ demonstrator action for the current
            state. This will be executed with probability `self.beta`.
            Otherwise, a "robot" action will be sampled and executed
            instead.

        Returns:
          next_obs, reward, done, info: unchanged output of `self.env.step()`.
        """
        assert self._is_reset, "call .reset() before .step()"

        # Replace the given action with a robot action 100*(1-beta)% of the time.
        if np.random.uniform(0, 1) > self.beta:
            actual_act = self.get_robot_act(self._last_obs)
        else:
            actual_act = user_action

        # actually step the env & record data as appropriate
        next_obs, reward, done, info = self.env.step(actual_act)
        self._last_obs = next_obs
        self.traj_accum.add_step(
            {"acts": user_action, "obs": next_obs, "rews": reward, "infos": info}
        )

        # if we're finished, then save the trajectory & print a message
        if done and not self._done_before:
            trajectory = self.traj_accum.finish_trajectory()
            timestamp = util.make_unique_timestamp()
            trajectory_path = os.path.join(
                self.save_dir, "dagger-demo-" + timestamp + ".npz"
            )
            logging.info(f"Saving demo at '{trajectory_path}'")
            _save_trajectory(trajectory_path, trajectory)

        if done:
            # record the fact that we're already done to avoid saving demo over and
            # over until the user resets
            self._done_before = True

        return next_obs, reward, done, info


class NeedsDemosException(Exception):
    """Signals demos need to be collected for current round before continuing."""


class DAggerTrainer:
    """Helper class for interactively training with DAgger.

    In essence, this is just BC with some helpers for incrementally
    resuming training and interpolating between demonstrator/learnt policies.
    Interaction proceeds in "rounds" in which the demonstrator first provides a
    fresh set of demonstrations, and then an underlying `BC` is invoked to
    fine-tune the policy on the entire set of demonstrations collected in all
    rounds so far. Demonstrations and policy/trainer checkpoints are stored in a
    directory with the following structure::

       scratch-dir-name/
           checkpoint-001.pkl
           checkpoint-002.pkl
           …
           checkpoint-XYZ.pkl
           checkpoint-latest.pkl
           demos/
               round-000/
                   demos_round_000_000.npz
                   demos_round_000_001.npz
                   …
               round-001/
                   demos_round_001_000.npz
                   …
               …
               round-XYZ/
                   …
    """

    SAVE_ATTRS = ("round_num",)
    DEMO_SUFFIX = ".npz"

    def __init__(
        self,
        env: gym.Env,
        scratch_dir: str,
        beta_schedule: Callable[[int], float] = None,
        batch_size: int = 32,
        **bc_kwargs,
    ):
        """Trainer constructor.

        Args:
            env: environment to train in.
            scratch_dir: directory to use to store intermediate training
                information (e.g. for resuming training).
            beta_schedule: provides a value of `beta` (the probability of taking
                expert action in any given state) at each round of training. If
                `None`, then `linear_beta_schedule` will be used instead.
            batch_size: Number of samples in each batch during BC training.
            **bc_kwargs: additional arguments for constructing the `BC` that
                will be used to train the underlying policy.
        """
        # for pickling
        self._init_args = locals()
        self._init_args.update(bc_kwargs)
        del self._init_args["self"]
        del self._init_args["bc_kwargs"]

        if beta_schedule is None:
            beta_schedule = LinearBetaSchedule(15)
        self.batch_size = batch_size
        self.beta_schedule = beta_schedule
        self.scratch_dir = scratch_dir
        self.env = env
        self.round_num = 0
        self.bc_kwargs = bc_kwargs
        self._last_loaded_round = -1
        self._all_demos = []

        self.bc_trainer = bc.BC(
            self.env.observation_space, self.env.action_space, **self.bc_kwargs
        )

    def _load_all_demos(self):
        num_demos_by_round = []
        for round_num in range(self._last_loaded_round + 1, self.round_num + 1):
            round_dir = self._demo_dir_path_for_round(round_num)
            demo_paths = self._get_demo_paths(round_dir)
            self._all_demos.extend(_load_trajectory(p) for p in demo_paths)
            num_demos_by_round.append(len(demo_paths))
        logging.info(f"Loaded {len(self._all_demos)} total")
        demo_transitions = rollout.flatten_trajectories(self._all_demos)
        return demo_transitions, num_demos_by_round

    def _get_demo_paths(self, round_dir):
        return [
            os.path.join(round_dir, p)
            for p in os.listdir(round_dir)
            if p.endswith(self.DEMO_SUFFIX)
        ]

    def _demo_dir_path_for_round(self, round_num=None):
        if round_num is None:
            round_num = self.round_num
        return os.path.join(self.scratch_dir, "demos", f"round-{round_num:03d}")

    def _check_has_latest_demos(self):
        demo_dir = self._demo_dir_path_for_round()
        demo_paths = self._get_demo_paths(demo_dir) if os.path.isdir(demo_dir) else []
        if len(demo_paths) == 0:
            raise NeedsDemosException(
                f"No demos found for round {self.round_num} in dir '{demo_dir}'. "
                f"Maybe you need to collect some demos? See "
                f".get_trajectory_collector()"
            )

    def _try_load_demos(self):
        self._check_has_latest_demos()
        if self._last_loaded_round < self.round_num:
            transitions, num_demos = self._load_all_demos()
            logging.info(
                f"Loaded {sum(num_demos)} new demos from {len(num_demos)} rounds"
            )
            data_loader = th_data.DataLoader(
                transitions,
                self.batch_size,
                drop_last=True,
                shuffle=True,
                collate_fn=types.transitions_collate_fn,
            )
            self.bc_trainer.set_expert_data_loader(data_loader)
            self._last_loaded_round = self.round_num

    def extend_and_update(self, **train_kwargs) -> int:
        """Extend internal batch of data and train.

        Specifically, this method will load new transitions (if necessary), train
        the model for a while, and advance the round counter. If there are no fresh
        demonstrations in the demonstration directory for the current round, then
        this will raise a `NeedsDemosException` instead of training or advancing
        the round counter. In that case, the user should call
        `.get_trajectory_collector()` and use the returned
        `InteractiveTrajectoryCollector` to produce a new set of demonstrations for
        the current interaction round.

        Arguments:
            **train_kwargs: arguments to pass to `BC.train()`.

        Returns:
            round_num: new round number after advancing the round counter.
        """
        logging.info("Loading demonstrations")
        self._try_load_demos()
        logging.info(f"Training at round {self.round_num}")
        self.bc_trainer.train(**train_kwargs)
        self.round_num += 1
        logging.info(f"New round number is {self.round_num}")
        return self.round_num

    def get_trajectory_collector(self) -> InteractiveTrajectoryCollector:
        """Create trajectory collector to extend current round's demonstration set.

        Returns:
            collector: an `InteractiveTrajectoryCollector` configured with the
                appropriate beta, appropriate imitator policy, etc. for the current
                round. Refer to the documentation for
                `InteractiveTrajectoryCollector` to see how to use this.
        """
        save_dir = self._demo_dir_path_for_round()
        beta = self.beta_schedule(self.round_num)

        def get_robot_act(obs):
            (
                (act,),
                _,
            ) = self.bc_trainer.policy.predict(obs[None])
            return act

        collector = InteractiveTrajectoryCollector(
            env=self.env, get_robot_act=get_robot_act, beta=beta, save_dir=save_dir
        )

        return collector

    def save_trainer(self) -> Tuple[str, str]:
        """Create a snapshot of trainer in the scratch/working directory.

        The created snapshot can be reloaded with `.reconstruct_trainer()`.
        In addition to saving one copy of the policy in the trainer snapshot, this
        method saves a second copy of the policy in its own file. Having a second copy
        of the policy is convenient because it can be loaded on its own and passed to
        evaluation routines for other algorithms.

        Returns:
            checkpoint_path: a path to one of the created `DAggerTrainer` checkpoints.
            policy_path: a path to one of the created `DAggerTrainer` policies.
        """
        os.makedirs(self.scratch_dir, exist_ok=True)

        # save full trainer checkpoints
        checkpoint_paths = [
            os.path.join(self.scratch_dir, f"checkpoint-{self.round_num:03d}.pt"),
            os.path.join(self.scratch_dir, "checkpoint-latest.pt"),
        ]
        for checkpoint_path in checkpoint_paths:
            th.save(self, checkpoint_path)

        # save policies separately for convenience
        policy_paths = [
            os.path.join(self.scratch_dir, f"policy-{self.round_num:03d}.pt"),
            os.path.join(self.scratch_dir, "policy-latest.pt"),
        ]
        for policy_path in policy_paths:
            self.save_policy(policy_path)

        return checkpoint_paths[0], policy_paths[0]

    def save_policy(self, policy_path: str) -> None:
        """Save the current policy only (and not the rest of the trainer).

        Args:
            policy_path: path to save policy to.
        """
        self.bc_trainer.save_policy(policy_path)
