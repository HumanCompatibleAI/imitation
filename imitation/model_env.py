"""Finite-horizon discrete environments with known transition dynamics. These
are handy when you want to perform exact maxent policy optimisation."""

import abc

import gym
from gym import spaces
import numpy as np


class ModelBasedEnv(gym.Env, metaclass=abc.ABCMeta):
    """ABC for tabular environments with known dynamics."""
    def __init__(self):
        self.cur_state = 0
        self.actions_taken = 0
        # we must cache action & observation spaces instead of reconstructing
        # anew so that the random state of the action space (!!) is preserved
        self._action_space = None
        self._observation_space = None
        self.seed()

    @property
    def action_space(self):
        if not self._action_space:
            self._action_space = spaces.Discrete(self.n_actions)
        return self._action_space

    @property
    def observation_space(self):
        if not self._observation_space:
            self._observation_space = spaces.Box(low=float('-inf'),
                                                 high=float('inf'),
                                                 shape=(self.obs_dim, ))
        return self._observation_space

    def seed(self, seed=None):
        if seed is None:
            seed = np.random.randint(0, 1 << 31)
        self.rand_state = np.random.RandomState(seed=seed)
        return [seed]

    def reset(self):
        self.cur_state = 0
        self.actions_taken = 0
        # as in step(), we copy so that it can't be mutated in-place (updates
        # will be reflected in self.observation_matrix!)
        return self.observation_matrix[self.cur_state].copy()

    def step(self, action):
        old_state = self.cur_state
        out_dist = self.transition_matrix[old_state, action]
        choice_states = np.arange(self.n_states)
        next_state = int(
            self.rand_state.choice(choice_states, p=out_dist, size=()))
        self.cur_state = next_state
        self.actions_taken += 1
        done = self.actions_taken >= self.horizon
        reward = self.reward_matrix[old_state]
        assert np.isscalar(reward), reward
        # copy so that it can't be mutated in-place
        obs = self.observation_matrix[next_state].copy()
        assert obs.ndim == 1, obs.shape
        infos = {"old_state": old_state, "new_state": next_state}
        return obs, reward, done, infos

    @property
    def n_states(self):
        """Number of states in this MDP (int)."""
        return self.transition_matrix.shape[0]

    @property
    def n_actions(self):
        """Number of actions in this MDP (int)."""
        return self.transition_matrix.shape[1]

    @property
    def obs_dim(self):
        """Size of observation vectors for this MDP."""
        return self.observation_matrix.shape[-1]

    # ############################### #
    # METHODS THAT MUST BE OVERRIDDEN #
    # ############################### #

    @property
    @abc.abstractmethod
    def transition_matrix(self):
        """Yield a 3D transition matrix with dimensions corresponding to
        current state, current action, and next state (in that order). In other
        words, if `T` is our returned matrix, then `T[s,a,sprime]` is the
        chance of transitioning into state `sprime` after taking action `a` in
        state `s`."""
        pass

    @property
    @abc.abstractmethod
    def observation_matrix(self):
        """Yields 2D observation matrix with dimensions corresponding to
        current state (first dim) and elements of observation (second dim)."""
        pass

    @property
    @abc.abstractmethod
    def reward_matrix(self):
        """Yields 2D reward matrix with dimensions corresponding to current
        state and current action."""
        pass

    @property
    @abc.abstractmethod
    def horizon(self):
        """Number of actions that can be taken in an episode."""
        pass


def make_random_trans_mat(
        n_states,
        n_actions,
        # maximum number of successors of an action in any
        # state
        max_branch_factor,
        # give an np.random.RandomState
        rand_state=np.random):
    """Make a 'random' transition matrix, in which each action goes to at least
    `max_branch-factor` other states from the current state, with transition
    distribution sampled from Dirichlet(1,1,…,1).

    This roughly apes the strategy from some old Lisp code that Rich Sutton
    left on the internet (http://incompleteideas.net/RandomMDPs.html), and is
    therefore a legitimate way to generate MDPs."""
    out_mat = np.zeros((n_states, n_actions, n_states), dtype='float32')
    state_array = np.arange(n_states)
    for start_state in state_array:
        for action in range(n_actions):
            # uniformly sample a number of successors in [1,max_branch_factor]
            # for this action
            succs = rand_state.randint(1, max_branch_factor + 1)
            next_states = rand_state.choice(state_array,
                                            size=(succs, ),
                                            replace=False)
            # generate random vec in probability simplex
            next_vec = rand_state.dirichlet(np.ones((succs, )))
            next_vec = next_vec / np.sum(next_vec)
            out_mat[start_state, action, next_states] = next_vec
    return out_mat


def make_obs_mat(
        n_states,
        # should we have random observations (True) or one-hot
        # observations (False)?
        is_random,
        # in case is_random==True: what should dimension of
        # observations be?
        obs_dim,
        # can pass in an np.random.RandomState if desired
        rand_state=np.random):
    """Make an observation matrix with a single observation for each state.
    Observations can either be drawn from random normal distribution (holds if
    `is_random=True`), or be unique one-hot vectors for each state."""
    if not is_random:
        assert obs_dim is None
    if is_random:
        obs_mat = rand_state.normal(0, 2, (n_states, obs_dim))
    else:
        obs_mat = np.identity(n_states)
    assert obs_mat.ndim == 2 \
        and obs_mat.shape[:1] == (n_states, ) \
        and obs_mat.shape[1] > 0
    return obs_mat


class RandomMDP(ModelBasedEnv):
    """An simple MDP with a random transition matrix (random in the sense of
    `make_random_trans_mat`)."""
    def __init__(self,
                 n_states,
                 n_actions,
                 branch_factor,
                 horizon,
                 random_obs,
                 *,
                 obs_dim=None,
                 generator_seed=None):
        super().__init__()
        if generator_seed is None:
            generator_seed = np.random.randint(0, 1 << 31)
        # this generator is ONLY for constructing the MDP, not for controlling
        # random outcomes during rollouts
        rand_gen = np.random.RandomState(seed=generator_seed)
        if random_obs:
            if obs_dim is None:
                obs_dim = n_states
        else:
            assert obs_dim is None
        self._observation_matrix = make_obs_mat(n_states=n_states,
                                                is_random=random_obs,
                                                obs_dim=obs_dim,
                                                rand_state=rand_gen)
        self._transition_matrix = make_random_trans_mat(
            n_states=n_states,
            n_actions=n_actions,
            max_branch_factor=branch_factor,
            rand_state=rand_gen)
        self._horizon = horizon
        self._reward_weights = rand_gen.randn(
            self._observation_matrix.shape[-1])
        # TODO: should I have action-dependent rewards? If so, how do I make
        # the reward function aware of the current action?
        self._reward_matrix = self._observation_matrix @ self._reward_weights
        assert self._reward_matrix.shape == (self.n_states, )

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


class CliffWorld(ModelBasedEnv):
    """
    A grid world like this:

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

    def __init__(self,
                 width,
                 height,
                 horizon,
                 use_xy_obs,
                 *,
                 rew_default=-1,
                 rew_goal=10,
                 rew_cliff=-10,
                 fail_p=0.3):
        super().__init__()
        assert width >= 3 and height >= 2, \
            "degenerate grid world requested; is this a bug?"
        self.width = width
        self.height = height
        succ_p = 1 - fail_p
        n_states = width * height
        O_mat = self._observation_matrix = np.zeros(
            (n_states, 2 if use_xy_obs else n_states))
        R_vec = self._reward_matrix = np.zeros((n_states, ))
        T_mat = self._transition_matrix = np.zeros((n_states, 4, n_states))
        self._horizon = horizon

        def to_id_clamp(row, col):
            """Convert (x,y) state to state ID, after clamp x & y to lie in
            grid."""
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
                        float(row) / (height - 1)
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

        assert np.allclose(np.sum(T_mat, axis=-1), 1, rtol=1e-5), \
            "un-normalised matrix %s" % O_mat

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

    def draw_value_vec(self, D):
        """Use matplotlib a vector of values for each state. The vector could
        represent things like reward, occupancy measure, etc."""
        import matplotlib.pyplot as plt
        grid = D.reshape(self.height, self.width)
        plt.imshow(grid)
        plt.gca().grid(False)
