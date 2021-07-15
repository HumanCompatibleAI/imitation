"""Test tabular environments and tabular MCE IRL."""

import gym
import numpy as np
import pytest

from imitation.algorithms.tabular_irl import (
    LinearRewardModel,
    MLPRewardModel,
    mce_irl,
    mce_occupancy_measures,
    mce_partition_fh,
)
from imitation.envs.examples.model_envs import RandomMDP
from imitation.envs.resettable_env import TabularModelEnv

JAX_IMPORT_FAIL = False
try:
    import jax.experimental.optimizers as jaxopt  # pytype: disable=import-error
except ImportError:  # pragma: no cover
    JAX_IMPORT_FAIL = True


skip_if_no_jax = pytest.mark.skipif(
    JAX_IMPORT_FAIL,
    reason=(
        "jax not installed (see imitation.algorithms.tabular_irl docstring for "
        "installation info)"
    ),
)


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
        while not done:
            act = env.action_space.sample()
            obs, rew, done, info = env.step(act)
            traj.append((obs, rew))
        rv.append(traj)
    return rv


@skip_if_no_jax
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
        obs_dim = (i * 3 + 4) ** 2 + i
        mdp = RandomMDP(
            n_states=n_states,
            n_actions=n_actions,
            branch_factor=branch_factor,
            horizon=horizon,
            random_obs=random_obs,
            obs_dim=obs_dim if random_obs else None,
            generator_seed=i,
        )

        # sanity checks on sizes of things
        assert mdp.transition_matrix.shape == (n_states, n_actions, n_states)
        assert np.allclose(1, np.sum(mdp.transition_matrix, axis=-1))
        assert np.all(mdp.transition_matrix >= 0)
        assert (
            mdp.observation_matrix.shape[0] == n_states
            and mdp.observation_matrix.ndim == 2
        )
        assert mdp.reward_matrix.shape == (n_states,)
        assert mdp.horizon == horizon
        assert np.all(mdp.initial_state_dist >= 0)
        assert np.allclose(1, np.sum(mdp.initial_state_dist))
        assert np.sum(mdp.initial_state_dist > 0) == branch_factor

        # make sure trajectories aren't all the same if we don't specify same
        # seed each time
        trajectories = rollouts(mdp, 100)
        assert len(set(map(str, trajectories))) > 1

        trajectories = rollouts(mdp, 100, seed=42)
        # make sure trajectories ARE all the same if we do specify the same
        # seed each time
        assert len(set(map(str, trajectories))) == 1


@skip_if_no_jax
def test_policy_om_random_mdp():
    """Test that optimal policy occupancy measure ("om") for a random MDP is sane."""
    mdp = gym.make("imitation/Random-v0")
    V, Q, pi = mce_partition_fh(mdp)
    assert np.all(np.isfinite(V))
    assert np.all(np.isfinite(Q))
    assert np.all(np.isfinite(pi))
    # Check it is a probability distribution along the last axis
    assert np.all(pi >= 0)
    assert np.allclose(np.sum(pi, axis=-1), 1)

    Dt, D = mce_occupancy_measures(mdp, pi=pi)
    assert np.all(np.isfinite(D))
    assert np.any(D > 0)
    # expected number of state visits (over all states) should be equal to the
    # horizon
    assert np.allclose(np.sum(D), mdp.horizon)


class ReasonableMDP(TabularModelEnv):
    observation_matrix = np.array(
        [
            [3, -5, -1, -1, -4, 5, 3, 0],
            # state 1 (top)
            [4, -4, 2, 2, -4, -1, -2, -2],
            # state 2 (bottom, equiv to top)
            [3, -1, 5, -1, 0, 2, -5, 2],
            # state 3 (middle, very low reward and so dominated by others)
            [-5, -1, 4, 1, 4, 1, 5, 3],
            # state 4 (final, all self loops, good reward)
            [2, -5, 1, -5, 1, 4, 4, -3],
        ]
    )
    transition_matrix = np.array(
        [
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
                [0, 0, 0, 0, 1],
            ],
            # transitions out of state 2 (basically the same)
            [
                # action 0: goes to state 3 or 4 (sub-optimal)
                [0, 0, 0, 0.05, 0.95],
                # action 1: goes to state 3 (bad)
                [0, 0, 0, 1, 0],
                # action 2: goes to state 4 (good!)
                [0, 0, 0, 0, 1],
            ],
            # transitions out of state 3 (all go to state 4)
            [
                # action 0
                [0, 0, 0, 0, 1],
                # action 1
                [0, 0, 0, 0, 1],
                # action 2
                [0, 0, 0, 0, 1],
            ],
            # transitions out of state 4 (all go back to state 0)
            [
                # action 0
                [1, 0, 0, 0, 0],
                # action 1
                [1, 0, 0, 0, 0],
                # action 2
                [1, 0, 0, 0, 0],
            ],
        ]
    )
    reward_matrix = np.array(
        [
            # state 0 (okay reward, but we can't go back so it doesn't matter)
            1,
            # states 1 & 2 have same (okay) reward
            2,
            2,
            # state 3 has very negative reward (so avoid it!)
            -20,
            # state 4 has pretty good reward (good enough that we should move out
            # of 1 & 2)
            3,
        ]
    )
    # always start in s0 or s4
    initial_state_dist = [0.5, 0, 0, 0, 0.5]
    horizon = 20


@skip_if_no_jax
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
    # check that Dt[0] matches our initial state dist
    assert np.allclose(Dt[0], mdp.initial_state_dist)


@pytest.mark.expensive
@pytest.mark.parametrize(
    "model_class,model_kwargs",
    [(LinearRewardModel, dict()), (MLPRewardModel, dict(hiddens=[32, 32]))],
)
@skip_if_no_jax
def test_mce_irl_reasonable_mdp(model_class, model_kwargs):
    # test MCE IRL on the MDP
    mdp = ReasonableMDP()

    # demo occupancy measure
    V, Q, pi = mce_partition_fh(mdp)
    Dt, D = mce_occupancy_measures(mdp, pi=pi)

    rmodel = model_class(mdp.obs_dim, seed=13, **model_kwargs)
    opt = jaxopt.adam(1e-2)
    final_weights, final_counts = mce_irl(mdp, opt, rmodel, D, linf_eps=1e-3)

    assert np.allclose(final_counts, D, atol=1e-3, rtol=1e-3)
    # make sure weights have non-insane norm
    assert np.linalg.norm(final_weights) < 1000
