"""Tests for DAgger."""

import contextlib
import glob
import os
import pickle

import numpy as np
import pytest
from stable_baselines3.common import policies

from imitation.algorithms import bc, dagger
from imitation.data import rollout
from imitation.policies import serialize
from imitation.util import util

ENV_NAME = "CartPole-v1"
EXPERT_POLICY_PATH = "tests/data/expert_models/cartpole_0/policies/final/"
EXPERT_ROLLOUTS_PATH = "tests/data/expert_models/cartpole_0/rollouts/final.pkl"


def test_beta_schedule():
    one_step_sched = dagger.LinearBetaSchedule(1)
    three_step_sched = dagger.LinearBetaSchedule(3)
    for i in range(10):
        assert np.allclose(one_step_sched(i), 1 if i == 0 else 0)
        assert np.allclose(three_step_sched(i), (3 - i) / 3 if i <= 2 else 0)


def test_traj_collector(tmpdir, venv):
    robot_calls = 0

    def get_random_acts(obs):
        nonlocal robot_calls
        robot_calls += len(obs)
        return [venv.action_space.sample() for _ in range(len(obs))]

    collector = dagger.InteractiveTrajectoryCollector(
        venv=venv, get_robot_acts=get_random_acts, beta=0.5, save_dir=tmpdir
    )
    collector.reset()
    zero_acts = np.zeros((venv.num_envs,), dtype="int")
    obs, rews, dones, infos = collector.step(zero_acts)
    assert np.all(rews != 0)
    assert not np.any(dones)
    for info in infos:
        assert isinstance(info, dict)
    # roll out ~5 * venv.num_envs episodes
    for i in range(1000):
        collector.step(zero_acts)

    # there is a <10^(-12) probability this fails by chance; we should be calling
    # robot with 50% prob each time
    assert 388 * venv.num_envs <= robot_calls <= 612 * venv.num_envs

    # All user/expert actions are zero. Therefore, all collected actions should be
    # zero.
    file_paths = glob.glob(os.path.join(tmpdir, "dagger-demo-*.npz"))
    assert len(file_paths) >= 5
    trajs = map(dagger._load_trajectory, file_paths)
    nonzero_acts = sum(np.sum(traj.acts != 0) for traj in trajs)
    assert nonzero_acts == 0


@pytest.fixture(params=[1, 4])
def num_envs(request):
    return request.param


@pytest.fixture
def venv(num_envs):
    return util.make_vec_env(ENV_NAME, num_envs)


@pytest.fixture
def expert_policy(venv):
    return serialize.load_policy("ppo", EXPERT_POLICY_PATH, venv)


@pytest.fixture(params=[True, False])
def expert_trajs(request):
    keep_trajs = request.param
    if keep_trajs:
        with open(EXPERT_ROLLOUTS_PATH, "rb") as f:
            return pickle.load(f)
    else:
        return None


def _build_dagger_trainer(tmpdir, venv, beta_schedule, expert_policy, expert_trajs):
    del expert_policy
    if expert_trajs is not None:
        pytest.skip(
            "DAggerTrainer does not use trajectories. "
            "Skipping to avoid duplicate test."
        )
    return dagger.DAggerTrainer(
        venv=venv,
        scratch_dir=tmpdir,
        beta_schedule=beta_schedule,
        bc_kwargs=dict(optimizer_kwargs=dict(lr=1e-3)),
    )


def _build_simple_dagger_trainer(
    tmpdir,
    venv,
    beta_schedule,
    expert_policy,
    expert_trajs,
):
    return dagger.SimpleDAggerTrainer(
        venv=venv,
        scratch_dir=tmpdir,
        beta_schedule=beta_schedule,
        bc_kwargs=dict(optimizer_kwargs=dict(lr=1e-3)),
        expert_policy=expert_policy,
        expert_trajs=expert_trajs,
    )


@pytest.fixture(params=[None, dagger.LinearBetaSchedule(1)])
def beta_schedule(request):
    return request.param


@pytest.fixture(params=[_build_dagger_trainer, _build_simple_dagger_trainer])
def init_trainer_fn(request, tmpdir, venv, beta_schedule, expert_policy, expert_trajs):
    # Provide a trainer initialization fixture in addition `trainer` fixture below
    # for tests that want to initialize multiple DAggerTrainer.
    trainer_fn = request.param
    return lambda: trainer_fn(tmpdir, venv, beta_schedule, expert_policy, expert_trajs)


@pytest.fixture
def trainer(init_trainer_fn):
    return init_trainer_fn()


@pytest.fixture
def simple_dagger_trainer(tmpdir, venv, beta_schedule, expert_policy, expert_trajs):
    return _build_simple_dagger_trainer(
        tmpdir, venv, beta_schedule, expert_policy, expert_trajs
    )


def test_trainer_needs_demos_exception_error(
    trainer,
    expert_trajs,
):
    assert trainer.round_num == 0
    error_ctx = pytest.raises(dagger.NeedsDemosException)
    if expert_trajs is not None and isinstance(trainer, dagger.SimpleDAggerTrainer):
        # In this case, demos should be preloaded and we shouldn't experience
        # the NeedsDemoException error.
        ctx = contextlib.nullcontext()
    else:
        # In all cases except the one above, an error should be raised because
        # there are no demos to update on.
        ctx = error_ctx

    with ctx:
        trainer.extend_and_update(dict(n_epochs=1))

    if ctx == contextlib.nullcontext():
        with error_ctx:
            # If ctx=nullcontext before, then we should fail on the second call
            # because there aren't any demos loaded into round 1 yet.
            trainer.extend_and_update(dict(n_epochs=1))


def test_trainer_makes_progress(trainer, venv, expert_policy):
    pre_train_rew_mean = rollout.mean_return(
        trainer.bc_trainer.policy,
        venv,
        sample_until=rollout.make_min_episodes(15),
        deterministic_policy=False,
    )
    # checking that the initial policy is poor can be flaky; sometimes the
    # randomly initialised policy performs very well, and it's not clear why
    # assert pre_train_rew_mean < 100
    for i in range(2):
        # roll out a few trajectories for dataset, then train for a few steps
        collector = trainer.get_trajectory_collector()
        for _ in range(5):
            obs = collector.reset()
            dones = [False] * venv.num_envs
            while not np.any(dones):
                expert_actions, _ = expert_policy.predict(obs, deterministic=True)
                obs, _, dones, _ = collector.step(expert_actions)
        trainer.extend_and_update(dict(n_epochs=1))
    # make sure we're doing better than a random policy would
    post_train_rew_mean = rollout.mean_return(
        trainer.bc_trainer.policy,
        venv,
        sample_until=rollout.make_min_episodes(15),
        deterministic_policy=True,
    )
    assert post_train_rew_mean - pre_train_rew_mean > 50, (
        f"pre-train mean {pre_train_rew_mean}, post-train mean "
        f"{post_train_rew_mean}"
    )


def test_trainer_save_reload(tmpdir, init_trainer_fn):
    trainer = init_trainer_fn()
    trainer.round_num = 3
    trainer.save_trainer()
    loaded_trainer = dagger.reconstruct_trainer(trainer.scratch_dir)
    assert loaded_trainer.round_num == trainer.round_num

    # old trainer and reloaded trainer should have same variable values
    old_vars = trainer.bc_trainer.policy.state_dict()
    new_vars = loaded_trainer.bc_trainer.policy.state_dict()
    assert len(new_vars) == len(old_vars)
    for var, values in new_vars.items():
        assert values.equal(old_vars[var])

    # also those values should be different from freshly initialized trainer
    third_trainer = init_trainer_fn()
    third_vars = third_trainer.bc_trainer.policy.state_dict()
    assert len(third_vars) == len(old_vars)
    assert not all(values.equal(old_vars[var]) for var, values in third_vars.items())


@pytest.mark.now
def test_simple_dagger_trainer_train(simple_dagger_trainer: dagger.SimpleDAggerTrainer):
    simple_dagger_trainer.train(total_timesteps=200, bc_train_kwargs=dict(n_batches=10))


def test_policy_save_reload(tmpdir, trainer):
    # just make sure the methods run; we already test them in test_bc.py
    policy_path = os.path.join(tmpdir, "policy.pt")
    trainer.save_policy(policy_path)
    pol = bc.reconstruct_policy(policy_path)
    assert isinstance(pol, policies.BasePolicy)
