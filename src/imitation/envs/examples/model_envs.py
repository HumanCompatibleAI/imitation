"""Example discrete MDPs for use with tabular MCE IRL."""

from typing import Optional

import gym
import numpy as np

from imitation.envs.resettable_env import TabularModelEnv


def make_random_trans_mat(
    n_states,
    n_actions,
    max_branch_factor,
    rand_state=np.random,
) -> np.ndarray:
    """Make a 'random' transition matrix.

    Each action goes to at least `max_branch_factor` other states from the
    current state, with transition distribution sampled from Dirichlet(1,1,…,1).

    This roughly apes the strategy from some old Lisp code that Rich Sutton
    left on the internet (http://incompleteideas.net/RandomMDPs.html), and is
    therefore a legitimate way to generate MDPs.

    Args:
        n_states: Number of states.
        n_actions: Number of actions.
        max_branch_factor: Maximum number of states that can be reached from
            each state-action pair.
        rand_state: NumPy random state.

    Returns:
        The transition matrix `mat`, where `mat[s,a,next_s]` gives the probability
        of transitioning to `next_s` after taking action `a` in state `s`.
    """
    out_mat = np.zeros((n_states, n_actions, n_states), dtype="float32")
    for start_state in range(n_states):
        for action in range(n_actions):
            # uniformly sample a number of successors in [1,max_branch_factor]
            # for this action
            succs = rand_state.randint(1, max_branch_factor + 1)
            next_states = rand_state.choice(n_states, size=(succs,), replace=False)
            # generate random vec in probability simplex
            next_vec = rand_state.dirichlet(np.ones((succs,)))
            next_vec = next_vec / np.sum(next_vec)
            out_mat[start_state, action, next_states] = next_vec
    return out_mat


def make_random_state_dist(
    n_avail: int,
    n_states: int,
    rand_state: np.random.RandomState = np.random,
) -> np.ndarray:
    """Make a random initial state distribution over n_states.

    Args:
        n_avail: Number of states available to transition into.
        n_states: Total number of states.
        rand_state: NumPy random state.

    Returns:
        An initial state distribution that is zero at all but a uniformly random
        chosen subset of `n_avail` states. This subset of chosen states are set to a
        sample from the uniform distribution over the (n_avail-1) simplex, aka the
        flat Dirichlet distribution.

    Raises:
        ValueError: If `n_avail` is not in the range `(0, n_states]`.
    """  # noqa: DAR402
    assert 0 < n_avail <= n_states
    init_dist = np.zeros((n_states,))
    next_states = rand_state.choice(n_states, size=(n_avail,), replace=False)
    avail_state_dist = rand_state.dirichlet(np.ones((n_avail,)))
    init_dist[next_states] = avail_state_dist
    assert np.sum(init_dist > 0) == n_avail
    init_dist = init_dist / np.sum(init_dist)
    return init_dist


def make_obs_mat(
    n_states: int,
    is_random: bool,
    obs_dim: Optional[int],
    rand_state: np.random.RandomState = np.random,
) -> np.ndarray:
    """Makes an observation matrix with a single observation for each state.

    Args:
        n_states (int): Number of states.
        is_random (bool): Are observations drawn at random?
                    If `True`, draw from random normal distribution.
                    If `False`, are unique one-hot vectors for each state.
        obs_dim (int or NoneType): Must be `None` if `is_random == False`.
                 Otherwise, this must be set to the size of the random vectors.
        rand_state (np.random.RandomState): Random number generator.

    Returns:
        A matrix of shape `(n_states, obs_dim if is_random else n_states)`.
    """
    if not is_random:
        assert obs_dim is None
    if is_random:
        obs_mat = rand_state.normal(0, 2, (n_states, obs_dim))
    else:
        obs_mat = np.identity(n_states)
    assert (
        obs_mat.ndim == 2 and obs_mat.shape[:1] == (n_states,) and obs_mat.shape[1] > 0
    )
    return obs_mat.astype(np.float32)


class RandomMDP(TabularModelEnv):
    """AN MDP with a random transition matrix.

    Random matrix is created by `make_random_trans_mat`.
    """

    def __init__(
        self,
        *,
        n_states: int,
        n_actions: int,
        branch_factor: int,
        horizon: int,
        random_obs: bool,
        obs_dim: Optional[int] = None,
        generator_seed: Optional[int] = None,
    ):
        """Builds RandomMDP.

        Args:
            n_states: Number of states.
            n_actions: Number of actions.
            branch_factor: Maximum number of states that can be reached from
                each state-action pair.
            horizon: The horizon of the MDP, i.e. the episode length.
            random_obs: Whether to use random observations (True)
                or one-hot coded (False).
            obs_dim: The size of the observation vectors; must be `None`
                if `random_obs == False`.
            generator_seed: Seed for NumPy RNG.
        """
        super().__init__()
        # this generator is ONLY for constructing the MDP, not for controlling
        # random outcomes during rollouts
        rand_gen = np.random.RandomState(generator_seed)
        if random_obs:
            if obs_dim is None:
                obs_dim = n_states
        else:
            assert obs_dim is None
        self._observation_matrix = make_obs_mat(
            n_states=n_states,
            is_random=random_obs,
            obs_dim=obs_dim,
            rand_state=rand_gen,
        )
        self._transition_matrix = make_random_trans_mat(
            n_states=n_states,
            n_actions=n_actions,
            max_branch_factor=branch_factor,
            rand_state=rand_gen,
        )
        self._initial_state_dist = make_random_state_dist(
            n_avail=branch_factor,
            n_states=n_states,
            rand_state=rand_gen,
        )
        self._horizon = horizon
        self._reward_weights = rand_gen.randn(self._observation_matrix.shape[-1])
        self._reward_matrix = self._observation_matrix @ self._reward_weights
        assert self._reward_matrix.shape == (self.n_states,)

    @property
    def observation_matrix(self):
        return self._observation_matrix

    @property
    def transition_matrix(self):
        return self._transition_matrix

    @property
    def reward_matrix(self):
        return self._reward_matrix

    @property
    def initial_state_dist(self):
        return self._initial_state_dist

    @property
    def horizon(self):
        return self._horizon


class CliffWorld(TabularModelEnv):
    """A grid world with a goal next to a cliff the agent may fall into.

    Illustration::

         0 1 2 3 4 5 6 7 8 9
        +-+-+-+-+-+-+-+-+-+-+  Wind:
      0 |S|C|C|C|C|C|C|C|C|G|
        +-+-+-+-+-+-+-+-+-+-+  ^ ^ ^
      1 | | | | | | | | | | |  | | |
        +-+-+-+-+-+-+-+-+-+-+
      2 | | | | | | | | | | |  ^ ^ ^
        +-+-+-+-+-+-+-+-+-+-+  | | |

    Aim is to get from S to G. The G square has reward +10, the C squares
    ("cliff") have reward -10, and all other squares have reward -1. Agent can
    move in all directions (except through walls), but there is 30% chance that
    they will be blown upwards by one more unit than intended due to wind.
    Optimal policy is to go out a bit and avoid the cliff, but still hit goal
    eventually.
    """

    def __init__(
        self,
        *,
        width: int,
        height: int,
        horizon: int,
        use_xy_obs: bool,
        rew_default: int = -1,
        rew_goal: int = 10,
        rew_cliff: int = -10,
        fail_p: float = 0.3,
    ):
        """Builds CliffWorld with specified dimensions and reward."""
        super().__init__()
        assert (
            width >= 3 and height >= 2
        ), "degenerate grid world requested; is this a bug?"
        self.width = width
        self.height = height
        succ_p = 1 - fail_p
        n_states = width * height
        O_mat = self._observation_matrix = np.zeros(
            (n_states, 2 if use_xy_obs else n_states),
            dtype=np.float32,
        )
        R_vec = self._reward_matrix = np.zeros((n_states,))
        T_mat = self._transition_matrix = np.zeros((n_states, 4, n_states))
        self._horizon = horizon

        def to_id_clamp(row, col):
            """Convert (x,y) state to state ID, after clamp x & y to lie in grid."""
            row = min(max(row, 0), height - 1)
            col = min(max(col, 0), width - 1)
            state_id = row * width + col
            assert 0 <= state_id < self.n_states
            return state_id

        for row in range(height):
            for col in range(width):
                state_id = to_id_clamp(row, col)

                # start by computing reward
                if row > 0:
                    r = rew_default  # blank
                elif col == 0:
                    r = rew_default  # start
                elif col == width - 1:
                    r = rew_goal  # goal
                else:
                    r = rew_cliff  # cliff
                R_vec[state_id] = r

                # now compute observation
                if use_xy_obs:
                    # (x, y) coordinate scaled to (0,1)
                    O_mat[state_id, :] = [
                        float(col) / (width - 1),
                        float(row) / (height - 1),
                    ]
                else:
                    # our observation matrix is just the identity; observation
                    # is an indicator vector telling us exactly what state
                    # we're in
                    O_mat[state_id, state_id] = 1

                # finally, compute transition matrix entries for each of the
                # four actions
                for drow in [-1, 1]:
                    for dcol in [-1, 1]:
                        action_id = (drow + 1) + (dcol + 1) // 2
                        target_state = to_id_clamp(row + drow, col + dcol)
                        fail_state = to_id_clamp(row + drow - 1, col + dcol)
                        T_mat[state_id, action_id, fail_state] += fail_p
                        T_mat[state_id, action_id, target_state] += succ_p

        assert np.allclose(np.sum(T_mat, axis=-1), 1, rtol=1e-5), (
            "un-normalised matrix %s" % O_mat
        )

    @property
    def observation_matrix(self):
        return self._observation_matrix

    @property
    def transition_matrix(self):
        return self._transition_matrix

    @property
    def reward_matrix(self):
        return self._reward_matrix

    @property
    def horizon(self):
        return self._horizon

    @property
    def initial_state_dist(self):
        # always start in s0
        rv = np.zeros((self.n_states,))
        rv[0] = 1.0
        return rv

    def draw_value_vec(self, D) -> None:
        """Use matplotlib to plot a vector of values for each state.

        The vector could represent things like reward, occupancy measure, etc.

        Args:
            D: the vector to plot.
        """
        import matplotlib.pyplot as plt

        grid = D.reshape(self.height, self.width)
        plt.imshow(grid)
        plt.gca().grid(False)


def register_cliff(suffix, kwargs):
    gym.register(
        f"imitation/CliffWorld{suffix}-v0",
        entry_point="imitation.envs.examples.model_envs:CliffWorld",
        kwargs=kwargs,
    )


for width, height, horizon in [(7, 4, 9), (15, 6, 18), (100, 20, 110)]:
    for use_xy in [False, True]:
        use_xy_str = "XY" if use_xy else ""
        register_cliff(
            f"{width}x{height}{use_xy_str}",
            kwargs={
                "width": width,
                "height": height,
                "use_xy_obs": use_xy,
                "horizon": horizon,
            },
        )

# These parameter choices are somewhat arbitrary.
# We anticipate most users will want to construct RandomMDP directly.
gym.register(
    "imitation/Random-v0",
    entry_point="imitation.envs.examples.model_envs:RandomMDP",
    kwargs={
        "n_states": 16,
        "n_actions": 3,
        "branch_factor": 2,
        "horizon": 20,
        "random_obs": True,
        "obs_dim": 5,
        "generator_seed": 42,
    },
)
