"""Test tabular environments and tabular MCE IRL."""

import numpy as np

from imitation.model_env import ModelBasedEnv, RandomMDP
from imitation.tabular_irl import (SGD, AMSGrad, maxent_irl,
                                   mce_occupancy_measures, mce_partition_fh,
                                   LinearRewardModel)

# import pytest


def rollouts(env, n=10, seed=None):
    rv = []
    for i in range(n):
        done = False
        if seed is not None:
            # if a seed is given, then we use the same seed each time (should
            # give same trajectory each time)
            env.seed(seed)
            env.action_space.seed(seed)
        obs = env.reset()
        traj = [obs]
        acts = []
        while not done:
            act = env.action_space.sample()
            acts.append(act)
            obs, rew, done, info = env.step(act)
            traj.append((obs, rew))
        rv.append(traj)
    return rv


def test_random_mdp():
    for i in range(3):
        n_states = 4 * (i + 3)
        n_actions = i + 2
        branch_factor = i + 1
        if branch_factor == 1:
            # make sure we have enough actions to get reasonable trajectories
            n_actions = min(n_states, max(n_actions, 4))
        horizon = 5 * (i + 1)
        random_obs = (i % 2) == 0
        obs_dim = (i * 3 + 4)**2 + i
        mdp = RandomMDP(
            n_states=n_states,
            n_actions=n_actions,
            branch_factor=branch_factor,
            horizon=horizon,
            random_obs=random_obs,
            obs_dim=obs_dim if random_obs else None,
            generator_seed=i)

        # sanity checks on sizes of things
        assert mdp.transition_matrix.shape == (n_states, n_actions, n_states)
        assert mdp.observation_matrix.shape[0] == n_states \
            and mdp.observation_matrix.ndim == 2

        # make sure trajectories aren't all the same if we don't specify same
        # seed each time
        trajectories = rollouts(mdp, 100)
        assert len(set(map(str, trajectories))) > 1

        trajectories = rollouts(mdp, 100, seed=42)
        # make sure trajectories ARE all the same if we do specify the same
        # seed each time
        assert len(set(map(str, trajectories))) == 1


def test_policy_om_random_mdp():
    mdp = RandomMDP(
        n_states=16,
        n_actions=3,
        branch_factor=2,
        horizon=20,
        random_obs=True,
        obs_dim=5,
        generator_seed=42)
    V, Q, pi = mce_partition_fh(mdp)
    assert np.all(np.isfinite(V))
    assert np.all(np.isfinite(Q))
    assert np.all(np.isfinite(pi))
    assert np.all(pi >= 0)
    # it (always?) has to take SOME actions
    assert np.any(pi > 0)
    # we do <= 1 instead of allclose() because pi actually doesn't have to be
    # normalised (per remark in Ziebart's thesis); missing probability mass
    # corresponds to early termination or something
    assert np.all(np.sum(pi, axis=-1) <= 1 + 1e-5)

    Dt, D = mce_occupancy_measures(mdp, pi=pi)
    assert np.all(np.isfinite(D))
    assert np.any(D > 0)
    # make sure we're in state 0 (the initial state) for at least one step, in
    # expectation
    assert D[0] >= 1
    # expected number of state visits (over all states) should be roughly equal
    # to the horizon
    assert np.sum(D) <= mdp.horizon + 1e-5
    # heuristic to make sure we're *roughly* having 90% of the state encounters
    # we expect we should
    assert np.sum(D) >= mdp.horizon * 0.9


class ReasonableMDP(ModelBasedEnv):
    observation_matrix = np.array([
        # TODO: remove first 5 features once I know this works for sure
        # state 0 (init)
        [1, 0, 0, 0, 0, 3, -5, -1, -1, -4, 5, 3, 0],
        # state 1 (top)
        [0, 1, 0, 0, 0, 4, -4, 2, 2, -4, -1, -2, -2],
        # state 2 (bottom, equiv to top)
        [0, 0, 1, 0, 0, 3, -1, 5, -1, 0, 2, -5, 2],
        # state 3 (middle, very low reward and so dominated by others)
        [0, 0, 0, 1, 0, -5, -1, 4, 1, 4, 1, 5, 3],
        # state 4 (final, all self loops, good reward)
        [0, 0, 0, 0, 1, 2, -5, 1, -5, 1, 4, 4, -3]
    ])
    transition_matrix = np.array([
        # transitions out of state 0
        [
            # action 0: goes to state 1 (sometimes 2)
            [0, 0.9, 0.1, 0, 0],
            # action 1: goes to state 3 deterministically
            [0, 0, 0, 1, 0],
            # action 2: goes to state 2 (sometimes 2)
            [0, 0.1, 0.9, 0, 0],
        ],
        # transitions out of state 1
        [
            # action 0: goes to state 3 or 4 (sub-optimal)
            [0, 0, 0, 0.05, 0.95],
            # action 1: goes to state 3 (bad)
            [0, 0, 0, 1, 0],
            # action 2: goes to state 4 (good!)
            [0, 0, 0, 0, 1]
        ],
        # transitions out of state 2 (basically the same)
        [
            # action 0: goes to state 3 or 4 (sub-optimal)
            [0, 0, 0, 0.05, 0.95],
            # action 1: goes to state 3 (bad)
            [0, 0, 0, 1, 0],
            # action 2: goes to state 4 (good!)
            [0, 0, 0, 0, 1]
        ],
        # transitions out of state 3 (all go to state 4)
        [
            # action 0
            [0, 0, 0, 0, 1],
            # action 1
            [0, 0, 0, 0, 1],
            # action 2
            [0, 0, 0, 0, 1]
        ],
        # transitions out of state 4 (all go back to state 0)
        [
            # action 0
            [1, 0, 0, 0, 0],
            # action 1
            [1, 0, 0, 0, 0],
            # action 2
            [1, 0, 0, 0, 0]
        ],
    ])
    reward_matrix = np.array([
        # state 0 (okay reward, but we can't go back so it doesn't matter)
        1,
        # states 1 & 2 have same (okay) reward
        2,
        2,
        # state 3 has very negative reward (so avoid it!)
        -20,
        # state 4 has pretty good reward (good enough that we should move out
        # of 1 & 2)
        3
    ])
    horizon = 20


def test_policy_om_reasonable_mdp():
    # MDP described above
    mdp = ReasonableMDP()
    # get policy etc. for our MDP
    V, Q, pi = mce_partition_fh(mdp)
    Dt, D = mce_occupancy_measures(mdp, pi=pi)
    assert np.all(np.isfinite(V))
    assert np.all(np.isfinite(Q))
    assert np.all(np.isfinite(pi))
    assert np.all(np.isfinite(Dt))
    assert np.all(np.isfinite(D))
    # check that actions 0 & 2 (which go to states 1 & 2) are roughly equal
    assert np.allclose(pi[:19, 0, 0], pi[:19, 0, 2])
    # also check that they're by far preferred to action 1 (that goes to state
    # 3, which has poor reward)
    assert np.all(pi[:19, 0, 0] > 2 * pi[:19, 0, 1])
    # make sure that states 3 & 4 have roughly uniform policies
    pi_34 = pi[:5, 3:5]
    assert np.allclose(pi_34, np.ones_like(pi_34) / 3.0)
    # check that states 1 & 2 have similar policies to each other
    assert np.allclose(pi[:19, 1, :], pi[:19, 2, :])
    # check that in state 1, action 2 (which goes to state 4 with certainty) is
    # better than action 0 (which only gets there with some probability), and
    # that both are better than action 1 (which always goes to the bad state).
    assert np.all(pi[:19, 1, 2] > pi[:19, 1, 0])
    assert np.all(pi[:19, 1, 0] > pi[:19, 1, 1])

    # now compute demonstrator features
    # initial reward weights
    rmodel = LinearRewardModel(mdp.obs_dim, seed=13)
    opt = AMSGrad(rmodel, alpha_sched=1e-2)
    final_weights, final_counts = maxent_irl(
        mdp, opt, rmodel, D, linf_eps=1e-2)


def test_optimisers():
    rng = np.random.RandomState(42)
    # make a positive definite Q
    Q = rng.randn(10, 10)
    Q = Q.T @ Q + 1e-5 * np.eye(10)
    v = rng.randn(10)

    def f(x):
        return 0.5 * x.T @ Q @ x + np.dot(v, x)

    def df(x):
        return Q @ x + v

    # also find the solution
    Qinv = np.linalg.inv(Q)
    solution = -Qinv @ v
    opt_value = f(solution)

    # start in some place far from minimum of f(x)
    sgd_rmodel = LinearRewardModel(10, seed=42)
    sgd = SGD(sgd_rmodel, alpha_sched=1e-2)
    # amsgrad typically requires (and can deal with) a higher step size
    agd_rmodel = LinearRewardModel(10, seed=42)
    agd = AMSGrad(agd_rmodel, alpha_sched=1e-1)
    for rmodel, optimiser in [(sgd_rmodel, sgd), (agd_rmodel, agd)]:
        x = rmodel.get_params()
        grad = df(x)
        val = f(x)
        assert np.linalg.norm(grad) > 1
        assert np.abs(val) > 1
        print('Initial: val=%.3f, grad=%.3f' % (val, np.linalg.norm(grad)))
        for it in range(25000):
            optimiser.step(grad)
            x = optimiser.current_params
            grad = df(x)
            # natural gradient: grad = Qinv @ x
            val = f(x)
            if 0 == (it % 50):
                print('Value %.3f (grad %.3f) after %d steps' %
                      (val, np.linalg.norm(grad), it))
            if np.linalg.norm(grad) < 1e-4:
                break
        # pretty loose because we use big step sizes
        assert np.linalg.norm(grad) < 1e-2
        assert np.sum(np.abs(x - solution)) < 1e-2
        assert np.abs(val - opt_value) < 1e-2
