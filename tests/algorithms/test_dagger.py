"""Tests for `imitation.algorithms.dagger`."""

import contextlib
import glob
import math
import os
from typing import List, Optional, Sequence
from unittest import mock

import gym
import numpy as np
import pytest
import torch.random
from stable_baselines3.common import evaluation, policies

from imitation.algorithms import bc, dagger
from imitation.data import rollout, types
from imitation.data.types import TrajectoryWithRew
from imitation.policies import base
from imitation.testing import reward_improvement
from imitation.util import util


@pytest.fixture(params=[True, False])
def maybe_pendulum_expert_trajectories(
    pendulum_expert_trajectories: Sequence[TrajectoryWithRew],
    request,
) -> Optional[Sequence[TrajectoryWithRew]]:
    keep_trajs = request.param
    if keep_trajs:
        return pendulum_expert_trajectories
    else:
        return None


def test_beta_schedule():
    one_step_sched = dagger.LinearBetaSchedule(1)
    three_step_sched = dagger.LinearBetaSchedule(3)
    for i in range(10):
        assert np.allclose(one_step_sched(i), 1 if i == 0 else 0)
        assert np.allclose(three_step_sched(i), (3 - i) / 3 if i <= 2 else 0)


def test_traj_collector_seed(tmpdir, pendulum_venv, rng):
    collector = dagger.InteractiveTrajectoryCollector(
        venv=pendulum_venv,
        get_robot_acts=lambda o: [
            pendulum_venv.action_space.sample() for _ in range(len(o))
        ],
        beta=0.5,
        save_dir=tmpdir,
        rng=rng,
    )
    seeds1 = collector.seed(42)
    obs1 = collector.reset()
    seeds2 = collector.seed(42)
    obs2 = collector.reset()

    np.testing.assert_array_equal(seeds1, seeds2)
    np.testing.assert_array_equal(obs1, obs2)


def test_traj_collector(tmpdir, pendulum_venv, rng):
    robot_calls = 0
    num_episodes = 0

    def get_random_acts(obs):
        nonlocal robot_calls
        robot_calls += len(obs)
        return [pendulum_venv.action_space.sample() for _ in range(len(obs))]

    collector = dagger.InteractiveTrajectoryCollector(
        venv=pendulum_venv,
        get_robot_acts=get_random_acts,
        beta=0.5,
        save_dir=tmpdir,
        rng=rng,
    )
    collector.reset()
    zero_acts = np.zeros(
        (pendulum_venv.num_envs,) + pendulum_venv.action_space.shape,
        dtype=pendulum_venv.action_space.dtype,
    )
    obs, rews, dones, infos = collector.step(zero_acts)
    assert np.all(rews != 0)
    assert not np.any(dones)
    for info in infos:
        assert isinstance(info, dict)
    # roll out 5 * venv.num_envs episodes (Pendulum-v1 has 200 timestep episodes)
    for i in range(1000):
        _, _, dones, _ = collector.step(zero_acts)
        num_episodes += np.sum(dones)

    # there is a <10^(-12) probability this fails by chance; we should be calling
    # robot with 50% prob each time
    assert 388 * pendulum_venv.num_envs <= robot_calls <= 612 * pendulum_venv.num_envs

    # All user/expert actions are zero. Therefore, all collected actions should be
    # zero.
    file_paths = glob.glob(os.path.join(tmpdir, "dagger-demo-*.npz"))
    assert num_episodes == 5 * pendulum_venv.num_envs
    assert len(file_paths) == num_episodes
    trajs = [types.load(p)[0] for p in file_paths]
    nonzero_acts = sum(np.sum(traj.acts != 0) for traj in trajs)
    assert nonzero_acts == 0


def _build_dagger_trainer(
    tmpdir,
    venv,
    beta_schedule,
    expert_policy,
    pendulum_expert_rollouts: List[TrajectoryWithRew],
    custom_logger,
    rng: np.random.Generator,
):
    del expert_policy
    if pendulum_expert_rollouts is not None:
        pytest.skip(
            "DAggerTrainer does not use trajectories. "
            "Skipping to avoid duplicate test.",
        )
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        optimizer_kwargs=dict(lr=1e-3),
        custom_logger=custom_logger,
        rng=rng,
    )
    return dagger.DAggerTrainer(
        venv=venv,
        scratch_dir=tmpdir,
        beta_schedule=beta_schedule,
        bc_trainer=bc_trainer,
        custom_logger=custom_logger,
        rng=rng,
    )


def _build_simple_dagger_trainer(
    tmpdir,
    venv,
    beta_schedule,
    expert_policy,
    pendulum_expert_rollouts: Optional[List[TrajectoryWithRew]],
    custom_logger,
    rng,
):
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        optimizer_kwargs=dict(lr=1e-3),
        custom_logger=custom_logger,
        rng=rng,
    )
    return dagger.SimpleDAggerTrainer(
        venv=venv,
        scratch_dir=tmpdir,
        beta_schedule=beta_schedule,
        bc_trainer=bc_trainer,
        expert_policy=expert_policy,
        expert_trajs=pendulum_expert_rollouts,
        custom_logger=custom_logger,
        rng=rng,
    )


@pytest.fixture(params=[None, dagger.LinearBetaSchedule(1)])
def beta_schedule(request):
    return request.param


@pytest.fixture(params=[_build_dagger_trainer, _build_simple_dagger_trainer])
def init_trainer_fn(
    request,
    tmpdir,
    pendulum_venv,
    beta_schedule,
    pendulum_expert_policy,
    maybe_pendulum_expert_trajectories: Optional[List[TrajectoryWithRew]],
    custom_logger,
    rng,
):
    # Provide a trainer initialization fixture in addition `trainer` fixture below
    # for tests that want to initialize multiple DAggerTrainer.
    trainer_fn = request.param
    return lambda: trainer_fn(
        tmpdir,
        pendulum_venv,
        beta_schedule,
        pendulum_expert_policy,
        maybe_pendulum_expert_trajectories,
        custom_logger,
        rng,
    )


@pytest.fixture
def trainer(init_trainer_fn):
    return init_trainer_fn()


@pytest.fixture
def simple_dagger_trainer(
    tmpdir,
    pendulum_venv,
    beta_schedule,
    pendulum_expert_policy,
    maybe_pendulum_expert_trajectories: Optional[List[TrajectoryWithRew]],
    custom_logger,
    rng,
):
    return _build_simple_dagger_trainer(
        tmpdir,
        pendulum_venv,
        beta_schedule,
        pendulum_expert_policy,
        maybe_pendulum_expert_trajectories,
        custom_logger,
        rng,
    )


def test_trainer_needs_demos_exception_error(
    trainer,
    maybe_pendulum_expert_trajectories: Optional[List[TrajectoryWithRew]],
):
    assert trainer.round_num == 0
    error_ctx = pytest.raises(dagger.NeedsDemosException)
    ctx: contextlib.AbstractContextManager
    if maybe_pendulum_expert_trajectories is not None and isinstance(
        trainer,
        dagger.SimpleDAggerTrainer,
    ):
        # In this case, demos should be preloaded and we shouldn't experience
        # the NeedsDemoException error.
        ctx = contextlib.nullcontext()
    else:
        # In all cases except the one above, an error should be raised because
        # there are no demos to update on.
        ctx = error_ctx

    with ctx:
        trainer.extend_and_update(dict(n_epochs=1))

    # If ctx==nullcontext before, then we should fail on the second call
    # because there aren't any demos loaded into round 1 yet.
    # If ctx==error_ctx, then still should fail once again on the second call.
    with error_ctx:
        trainer.extend_and_update(dict(n_epochs=1))


def test_trainer_train_arguments(trainer, pendulum_expert_policy, rng):
    def add_samples():
        collector = trainer.create_trajectory_collector()
        rollout.generate_trajectories(
            pendulum_expert_policy,
            collector,
            sample_until=rollout.make_min_timesteps(40),
            rng=rng,
        )

    # Lower default number of epochs for the no-arguments call that follows.
    add_samples()
    with mock.patch.object(trainer, "DEFAULT_N_EPOCHS", 1):
        trainer.extend_and_update()

    add_samples()
    trainer.extend_and_update(dict(n_batches=2))

    add_samples()
    trainer.extend_and_update(dict(n_epochs=1))


def test_trainer_makes_progress(init_trainer_fn, pendulum_venv, pendulum_expert_policy):
    with torch.random.fork_rng():
        # manually seed to avoid flakiness
        torch.random.manual_seed(42)
        pendulum_venv.action_space.seed(42)

        trainer = init_trainer_fn()
        novice_rewards, _ = evaluation.evaluate_policy(
            trainer.policy,
            pendulum_venv,
            15,
            deterministic=False,
            return_episode_rewards=True,
        )
        # note a randomly initialised policy does well for some seeds -- so may
        # want to adjust this check if changing seed. Pendulum return can range
        # from -1,200 to -130 (approx.), per Figure 3 in this PDF (on page 3):
        # https://arxiv.org/pdf/2106.09556.pdf
        assert np.mean(novice_rewards) < -1000
        # Train for 10 iterations. (6 or less causes test to fail on some configs.)
        # see https://github.com/HumanCompatibleAI/imitation/issues/580 for details
        for i in range(10):
            # roll out a few trajectories for dataset, then train for a few steps
            collector = trainer.create_trajectory_collector()
            for _ in range(5):
                obs = collector.reset()
                dones = [False] * pendulum_venv.num_envs
                while not np.any(dones):
                    expert_actions, _ = pendulum_expert_policy.predict(
                        obs,
                        deterministic=True,
                    )
                    obs, _, dones, _ = collector.step(expert_actions)
            trainer.extend_and_update(dict(n_epochs=1))
        # make sure we're doing better than a random policy would
        rewards_after_training, _ = evaluation.evaluate_policy(
            trainer.policy,
            pendulum_venv,
            15,
            return_episode_rewards=True,
        )

    assert reward_improvement.is_significant_reward_improvement(
        novice_rewards,
        rewards_after_training,
    )
    assert reward_improvement.mean_reward_improved_by(
        novice_rewards,
        rewards_after_training,
        300,
    )


def test_trainer_save_reload(tmpdir, init_trainer_fn, pendulum_venv):
    trainer = init_trainer_fn()
    trainer.round_num = 3
    trainer.save_trainer()
    loaded_trainer = dagger.reconstruct_trainer(trainer.scratch_dir, venv=pendulum_venv)
    assert loaded_trainer.round_num == trainer.round_num

    # old trainer and reloaded trainer should have same variable values
    old_vars = trainer.policy.state_dict()
    new_vars = loaded_trainer.policy.state_dict()
    assert len(new_vars) == len(old_vars)
    assert all(values.equal(old_vars[var]) for var, values in new_vars.items())

    # also those values should be different from freshly initialized trainer
    third_trainer = init_trainer_fn()
    third_vars = third_trainer.policy.state_dict()
    assert len(third_vars) == len(old_vars)
    assert not all(values.equal(old_vars[var]) for var, values in third_vars.items())


@pytest.mark.parametrize("num_episodes", [1, 4])
def test_simple_dagger_trainer_train(
    simple_dagger_trainer: dagger.SimpleDAggerTrainer,
    pendulum_venv,
    num_episodes: int,
    tmpdir: str,
):
    episode_length = 200  # for Pendulum-v1
    rollout_min_episodes = 2
    simple_dagger_trainer.train(
        total_timesteps=episode_length * num_episodes,
        bc_train_kwargs=dict(n_batches=10),
        rollout_round_min_episodes=rollout_min_episodes,
        rollout_round_min_timesteps=1,
    )

    episodes_per_round = max(rollout_min_episodes, pendulum_venv.num_envs)
    num_rounds = math.ceil(num_episodes / episodes_per_round)

    round_paths = glob.glob(os.path.join(tmpdir, "demos", "round-*"))
    assert len(round_paths) == num_rounds
    for directory in round_paths:
        file_paths = glob.glob(os.path.join(directory, "dagger-demo-*.npz"))
        assert len(file_paths) == episodes_per_round


def test_policy_save_reload(tmpdir, trainer):
    # just make sure the methods run; we already test them in test_bc.py
    policy_path = os.path.join(tmpdir, "policy.pt")
    trainer.save_policy(policy_path)
    pol = bc.reconstruct_policy(policy_path)
    assert isinstance(pol, policies.BasePolicy)


def test_simple_dagger_space_mismatch_error(
    tmpdir,
    pendulum_venv,
    beta_schedule,
    pendulum_expert_policy,
    maybe_pendulum_expert_trajectories: Optional[List[TrajectoryWithRew]],
    custom_logger,
    rng,
):
    class MismatchedSpace(gym.spaces.Space):
        """Dummy space that is not equal to any other space."""

    # Swap out expert_policy.{observation,action}_space with a bad space to
    # elicit space mismatch errors.
    space = MismatchedSpace()
    for space_name in ["observation", "action"]:
        with mock.patch.object(pendulum_expert_policy, f"{space_name}_space", space):
            with pytest.raises(ValueError, match=f"Mismatched {space_name}.*"):
                _build_simple_dagger_trainer(
                    tmpdir,
                    pendulum_venv,
                    beta_schedule,
                    pendulum_expert_policy,
                    maybe_pendulum_expert_trajectories,
                    custom_logger,
                    rng,
                )


def test_dagger_not_enough_transitions_error(tmpdir, custom_logger, rng):
    venv = util.make_vec_env("CartPole-v0", rng=rng)
    # Initialize with large batch size to ensure error down the line.
    bc_trainer = bc.BC(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        batch_size=100_000,
        custom_logger=custom_logger,
        rng=rng,
    )
    trainer = dagger.DAggerTrainer(
        venv=venv,
        scratch_dir=tmpdir,
        bc_trainer=bc_trainer,
        custom_logger=custom_logger,
        rng=rng,
    )
    collector = trainer.create_trajectory_collector()
    policy = base.RandomPolicy(venv.observation_space, venv.action_space)
    rollout.generate_trajectories(
        policy,
        collector,
        rollout.make_min_episodes(1),
        rng=rng,
    )
    with pytest.raises(ValueError, match="Not enough transitions.*"):
        trainer.extend_and_update()
