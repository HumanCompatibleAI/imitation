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
import pathlib
from typing import Callable, List, Optional, Tuple, Union

import numpy as np
import torch as th
from stable_baselines3.common import logger, policies, utils, vec_env
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
    scratch_dir: types.AnyPath, device: Union[th.device, str] = "auto"
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
    checkpoint_path = pathlib.Path(scratch_dir, "checkpoint-latest.pt")
    return th.load(checkpoint_path, map_location=utils.get_device(device))


def _save_dagger_demo(
    trajectory: types.Trajectory,
    save_dir: types.AnyPath,
    prefix: str = "",
) -> None:
    save_dir = pathlib.Path(save_dir)
    assert isinstance(trajectory, types.Trajectory)
    actual_prefix = f"{prefix}-" if prefix else ""
    timestamp = util.make_unique_timestamp()
    filename = f"{actual_prefix}dagger-demo-{timestamp}.npz"

    save_dir.mkdir(parents=True, exist_ok=True)
    npz_path = pathlib.Path(save_dir, filename)
    np.savez_compressed(npz_path, **dataclasses.asdict(trajectory))
    logging.info(f"Saved demo at '{npz_path}'")


def _load_trajectory(npz_path: str) -> types.Trajectory:
    """Load a single trajectory from a compressed Numpy file."""
    np_data = np.load(npz_path, allow_pickle=True)
    has_rew = "rews" in np_data
    dict_data = dict(np_data.items())

    # infos=None is saved as array(None) which leads to a type checking error upon
    # `Trajectory` initialization. Convert to None to prevent error.
    infos = dict_data["infos"]
    if infos.shape == ():
        assert infos.item() is None
        dict_data["infos"] = None

    cls = types.TrajectoryWithRew if has_rew else types.Trajectory
    return cls(**dict_data)


class InteractiveTrajectoryCollector(vec_env.VecEnvWrapper):
    """DAgger VecEnvWrapper for querying and saving expert actions.

    Every call to `.step(actions)` accepts and saves expert actions to `self.save_dir`,
    but only forwards expert actions to the wrapped VecEnv with probability
    `self.beta`. With probability `1 - self.beta`, a "robot" action (i.e
    an action from the imitation policy) is forwarded instead.

    Demonstrations are saved as `TrajectoryWithRew` to `self.save_dir` at the end
    of every episode.
    """

    # TODO(shwang): Consider adding a Callable param that determines whether we
    #   query the expert for input or just automatically step() by ourselves using
    #   the robot action.

    def __init__(
        self,
        venv: vec_env.VecEnv,
        get_robot_acts: Callable[[np.ndarray], np.ndarray],
        beta: float,
        save_dir: types.AnyPath,
    ):
        """Trajectory collector constructor.

        Args:
          venv: vectorized environment to sample trajectories from.
          get_robot_acts: get robot actions that can be substituted for
              human actions. Takes a vector of observations as input & returns a
              vector of actions.
          beta: fraction of the time to use action given to .step() instead of
              robot action.
          save_dir: directory to save collected trajectories in.
        """
        super().__init__(venv)
        self.get_robot_acts = get_robot_acts
        assert 0 <= beta <= 1
        self.beta = beta
        self.traj_accum = None
        self.save_dir = save_dir
        self._last_obs = None
        self._done_before = True
        self._is_reset = False
        self._last_user_actions = None

    def reset(self) -> np.ndarray:
        """Resets the environment.

        Returns:
            obs: first observation of a new trajectory.
        """
        self.traj_accum = rollout.TrajectoryAccumulator()
        obs = self.venv.reset()
        for i, ob in enumerate(obs):
            self.traj_accum.add_step({"obs": ob}, key=i)
        self._last_obs = obs
        self._is_reset = True
        self._last_user_actions = None
        return obs

    def step_async(self, actions: np.ndarray) -> None:
        """Steps with a `1 - beta` chance of using `self.get_robot_acts` instead.

        DAgger needs to be able to inject imitation policy actions randomly at some
        subset of time steps. This method has a `self.beta` chance of keeping the
        `actions` passed in as an argument, and a `1 - self.beta` chance of
        will forwarding an actions generated by `self.get_robot_acts` instead.
        "robot" (i.e. imitation policy) action if necessary.

        At the end of every episode, a `TrajectoryWithRew` is saved to `self.save_dir`,
        where every saved action is the expert action, regardless of whether the
        robot action was used during that timestep.

        Args:
          actions: the _intended_ demonstrator/expert actions for the current
            state. This will be executed with probability `self.beta`.
            Otherwise, a "robot" (typically a BC policy) action will be sampled
            and executed instead via `self.get_robot_act`.

        Returns:
          next_obs, reward, done, info: unchanged output of `self.env.step()`.
        """
        assert self._is_reset, "call .reset() before .step()"

        # Replace the given action with a robot action 100*(1-beta)% of the time.
        if np.random.uniform(0, 1) > self.beta:
            actual_acts = self.get_robot_acts(self._last_obs)
        else:
            actual_acts = actions

        self._last_user_actions = actions
        self.venv.step_async(actual_acts)

    def step_wait(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray, dict]:
        next_obs, rews, dones, infos = self.venv.step_wait()
        self._last_obs = next_obs
        fresh_demos = self.traj_accum.add_steps_and_auto_finish(
            obs=next_obs,
            acts=self._last_user_actions,
            rews=rews,
            infos=infos,
            dones=dones,
        )
        for traj in fresh_demos:
            _save_dagger_demo(traj, self.save_dir)

        return next_obs, rews, dones, infos

    def flush_trajectories(self) -> None:
        """Immediately save all partially completed trajectories to `self.save_dir`."""
        # TODO(shwang): Since we are saving partial trajectories instead of
        #  whole trajectories, it might make sense to just transition
        #  to saving TransitionsMinimal. One easy way to do this could be via
        #  `imitation.data.wrappers.BufferingWrapper.pop_transitions`.
        #
        #  Ah yes, looks like we use Transitions anyways in the
        #  `DAggerTrainer._try_load_demos` stage.
        for i in range(self.num_envs):
            if len(self.traj_accum.partial_trajectories[i]) >= 1:
                traj = self.traj_accum.finish_trajectory(i)
                _save_dagger_demo(traj, self.save_dir)


class NeedsDemosException(Exception):
    """Signals demos need to be collected for current round before continuing."""


class DAggerTrainer:
    """DAgger training class with low-level API suitable for interactive human feedback.

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
        venv: vec_env.VecEnv,
        scratch_dir: types.AnyPath,
        beta_schedule: Callable[[int], float] = None,
        batch_size: int = 32,
        bc_kwargs: Optional[dict] = None,
    ):
        """DaggerTrainer constructor.

        Args:
            venv: Vectorized training environment. Note that when the robot
                action is randomly injected (in accordance with `beta_schedule`
                argument), every individual environment will get a robot action
                simulatenously for that timestep.
            scratch_dir: Directory to use to store intermediate training
                information (e.g. for resuming training).
            beta_schedule: Provides a value of `beta` (the probability of taking
                expert action in any given state) at each round of training. If
                `None`, then `linear_beta_schedule` will be used instead.
            batch_size: Number of samples in each batch during BC training.
            bc_kwargs: Additional arguments for constructing the `BC` that
                will be used to train the underlying policy.
        """
        # for pickling
        # self._init_args = locals()
        # self._init_args.update(bc_kwargs)
        # del self._init_args["self"]
        # del self._init_args["bc_kwargs"]

        if beta_schedule is None:
            beta_schedule = LinearBetaSchedule(15)
        self.batch_size = batch_size
        self.beta_schedule = beta_schedule
        self.scratch_dir = pathlib.Path(scratch_dir)
        self.venv = venv
        self.round_num = 0
        self.bc_kwargs = bc_kwargs or {}
        self._last_loaded_round = -1
        self._all_demos = []

        self.bc_trainer = bc.BC(
            self.venv.observation_space,
            self.venv.action_space,
            **self.bc_kwargs,
        )

    @property
    def policy(self) -> policies.BasePolicy:
        return self.bc_trainer.policy

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

    def _demo_dir_path_for_round(self, round_num: Optional[int] = None) -> pathlib.Path:
        if round_num is None:
            round_num = self.round_num
        return self.scratch_dir / "demos" / f"round-{round_num:03d}"

    def _try_load_demos(self) -> None:
        """Load the dataset for this round into self.bc_trainer as a DataLoader."""
        demo_dir = self._demo_dir_path_for_round()
        demo_paths = self._get_demo_paths(demo_dir) if os.path.isdir(demo_dir) else []
        if len(demo_paths) == 0:
            raise NeedsDemosException(
                f"No demos found for round {self.round_num} in dir '{demo_dir}'. "
                f"Maybe you need to collect some demos? See "
                f".get_trajectory_collector()"
            )

        if self._last_loaded_round < self.round_num:
            # TODO(shwang): Do we actually only want to load the most recent round's
            #   data, or should we load all the data?
            transitions, num_demos = self._load_all_demos()
            logging.info(
                f"Loaded {sum(num_demos)} new demos from {len(num_demos)} rounds"
            )
            data_loader = th_data.DataLoader(
                transitions,
                self.batch_size,
                # If drop_last=True, then BC.train() fails on "no batches available"
                # when len(transitions) < self.batch_size.
                drop_last=False,
                shuffle=True,
                collate_fn=types.transitions_collate_fn,
            )
            self.bc_trainer.set_expert_data_loader(data_loader)
            self._last_loaded_round = self.round_num

    def _default_bc_train_kwargs(self) -> dict:
        """Get default kwargs used for calls to `self.bc_trainer.train`.

        Note that these defaults set the `log_rollouts_venv` argument, meaning that
        `BC.train` will periodically roll out episodes to log the average BC policy
        return.
        """
        return dict(
            n_epochs=4,
            log_rollouts_venv=self.venv,
        )

    def extend_and_update(self, bc_train_kwargs: Optional[dict]) -> int:
        """Extend internal batch of data and train BC.

        Specifically, this method will load new transitions (if necessary), train
        the model for a while, and advance the round counter. If there are no fresh
        demonstrations in the demonstration directory for the current round, then
        this will raise a `NeedsDemosException` instead of training or advancing
        the round counter. In that case, the user should call
        `.get_trajectory_collector()` and use the returned
        `InteractiveTrajectoryCollector` to produce a new set of demonstrations for
        the current interaction round.

        Arguments:
            bc_train_kwargs: A dictionary of keyword arguments to pass to
                `BC.train()`, or None to use `self._default_bc_train_kwargs()`.

        Returns:
            round_num: new round number after advancing the round counter.
        """
        if bc_train_kwargs is None:
            bc_train_kwargs = self._default_bc_train_kwargs()
        logging.info("Loading demonstrations")
        self._try_load_demos()
        logging.info(f"Training at round {self.round_num}")
        self.bc_trainer.train(**bc_train_kwargs)
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
        collector = InteractiveTrajectoryCollector(
            venv=self.venv,
            get_robot_acts=lambda acts: self.bc_trainer.policy.predict(acts)[0],
            beta=beta,
            save_dir=save_dir,
        )
        return collector

    def save_trainer(self) -> Tuple[pathlib.Path, pathlib.Path]:
        """Create a snapshot of trainer in the scratch/working directory.

        The created snapshot can be reloaded with `reconstruct_trainer()`.
        In addition to saving one copy of the policy in the trainer snapshot, this
        method saves a second copy of the policy in its own file. Having a second copy
        of the policy is convenient because it can be loaded on its own and passed to
        evaluation routines for other algorithms.

        Returns:
            checkpoint_path: a path to one of the created `DAggerTrainer` checkpoints.
            policy_path: a path to one of the created `DAggerTrainer` policies.
        """
        self.scratch_dir.mkdir(parents=True, exist_ok=True)

        # save full trainer checkpoints
        checkpoint_paths = [
            self.scratch_dir / f"checkpoint-{self.round_num:03d}.pt",
            self.scratch_dir / "checkpoint-latest.pt",
        ]
        for checkpoint_path in checkpoint_paths:
            th.save(self, checkpoint_path)

        # save policies separately for convenience
        policy_paths = [
            self.scratch_dir / f"policy-{self.round_num:03d}.pt",
            self.scratch_dir / "policy-latest.pt",
        ]
        for policy_path in policy_paths:
            self.save_policy(policy_path)

        return checkpoint_paths[0], policy_paths[0]

    def save_policy(self, policy_path: types.AnyPath) -> None:
        """Save the current policy only (and not the rest of the trainer).

        Args:
            policy_path: path to save policy to.
        """
        self.bc_trainer.save_policy(policy_path)


class SimpleDAggerTrainer(DAggerTrainer):
    """Simpler subclass of DAggerTrainer for training with synthetic feedback."""

    def __init__(
        self,
        venv: vec_env.VecEnv,
        log_dir: types.AnyPath,
        expert_policy: policies.BasePolicy,
        expert_trajs: Optional[List[types.Trajectory]] = None,
        beta_schedule: Callable[[int], float] = None,
        batch_size: int = 32,
        bc_kwargs: Optional[dict] = None,
    ):
        """SimpleDAggerTrainer constructor.
        Args:
            venv: Vectorized training environment. Note that when the robot
                action is randomly injected (in accordance with `beta_schedule`
                argument), every individual environment will get a robot action
                simulatenously for that timestep.
            log_dir: Directory for storing Tensorboard logs, policies, checkpoints,
                and demonstrations.
            expert_policy: The expert policy used to generate synthetic demonstrations.
            expert_trajs: Optional starting dataset that is inserted into the round 0
                dataset.
            batch_size: Number of samples in each batch during BC training.
            bc_kwargs: additional arguments for constructing the `BC` that
                will be used to train the underlying policy.
        """
        self.log_dir = pathlib.Path(log_dir)
        super().__init__(
            venv=venv,
            scratch_dir=self.log_dir / "scratch",
            beta_schedule=beta_schedule,
            batch_size=batch_size,
            bc_kwargs=bc_kwargs,
        )
        self.expert_policy = expert_policy
        if expert_policy.observation_space != venv.observation_space:
            raise ValueError(
                "Mismatched observation space between expert_policy and env"
            )
        if expert_policy.action_space != venv.action_space:
            raise ValueError("Mismatched action space between expert_policy and env")

        # TODO(shwang):
        #   Might welcome Transitions and DataLoaders as sources of expert data
        #   in the future too, but this will require some refactoring, so for
        #   now we just have `expert_trajs`.
        if expert_trajs is not None:
            # Save each initial expert trajectory into the "round 0" demonstration
            # data directory.
            for traj in expert_trajs:
                _save_dagger_demo(
                    traj,
                    self._demo_dir_path_for_round(),
                    prefix="initial_data",
                )

    def train(
        self,
        total_timesteps: int,
        *,
        rollout_round_min_episodes: int = 3,
        rollout_round_min_timesteps: int = 500,
        bc_train_kwargs: Optional[dict] = None,
    ) -> None:
        """Train the DAgger agent.

        The agent is trained in "rounds" where each round consists of a dataset
        aggregation step followed by BC update step.

        During a dataset aggregation step, `self.expert_policy` is used to perform
        rollouts in the environment but there is a `1 - beta` chance (beta is
        determined from the round number and `self.beta_schedule`) that the DAgger
        agent's action is used instead. Regardless of whether the DAgger agent's action
        is used during the rollout, the expert action and corresponding observation are
        always appended to the dataset. The number of environment steps in the
        dataset aggregation stage is determined by the `rollout_round_min*` arguments.

        During a BC update step, `BC.train()` is called to update the DAgger agent on
        the dataset collected in this round.

        Args:
            total_timesteps: The number of timesteps to train inside the environment.
                (In practice this is a lower bound, as the number of timesteps is
                rounded up to finish DAgger training rounds, and the
                environment timesteps are executed in multiples of
                `self.venv.num_envs`.)
            rollout_round_min_episodes: The number of episodes the must be completed
                completed before a dataset aggregation step ends.
            rollout_round_min_timesteps: The number of environment timesteps that must
                be completed before a dataset aggregation step ends.
            bc_train_kwargs: Keyword arguments to pass to `BC.train` during the
                BC update step. If this is None, then `self._default_bc_train_kwargs()`
                is used.
        """
        total_timestep_count = 0
        round_num = 0

        while total_timestep_count < total_timesteps:
            collector = self.get_trajectory_collector()
            round_episode_count = 0
            round_timestep_count = 0

            obs = collector.reset()
            ep_rewards = np.zeros([self.venv.num_envs])
            # TODO(shwang): This while loop end condition causes rollout collection to
            #   suffer from the same problem that
            #   `imitation.data.rollout.generate_trajectories` previously had -- a
            #   bias towards shorter episodes.
            #   We could probably solve this problem simply by calling
            #   `generate_trajectories(self.expert_policy, venv=collector)` now that
            #   we are ported from using `Env` to `VecEnv`?
            while (
                round_episode_count < rollout_round_min_episodes
                and round_timestep_count < rollout_round_min_timesteps
            ):
                acts, _ = self.expert_policy.predict(obs, deterministic=True)
                obs, rews, dones, _ = collector.step(acts)
                total_timestep_count += self.venv.num_envs
                round_timestep_count += self.venv.num_envs
                ep_rewards += rews

                for i, done in enumerate(dones):
                    if done:
                        logger.record_mean("dagger/mean_episode_reward", ep_rewards[i])
                        ep_rewards[i] = 0
                round_episode_count += 1

            # Flush partial trajectories so that any timesteps before episode
            # completion are also available in the dataset.
            # Without this call, self.extend_and_update() currently
            # fails if the "round" also didn't finish any episodes (not enough
            # timesteps).
            collector.flush_trajectories()

            logger.record("dagger/total_timesteps", total_timestep_count)
            logger.record("dagger/round_num", round_num)
            logger.record("dagger/round_episode_count", round_episode_count)
            logger.record("dagger/round_timestep_count", round_timestep_count)

            # TODO(shwang): It looks like BC might start looping Tensorboard
            #   back to x=0 with each new call to BC.train(). Consider adding a
            #   `reset_tensorboard: bool = False` argument to BC.train() if this turns
            #   out to be the case?

            # `logger.dump` is called inside BC.train within the following fn call:
            self.extend_and_update(bc_train_kwargs)
            round_num += 1
