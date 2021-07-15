"""Finite-horizon tabular Maximum Causal Entropy IRL.

Follows the description in chapters 9 and 10 of Brian Ziebart's `PhD thesis`_.

Uses NumPy-based optimizer Jax, so the code can be run without
PyTorch/TensorFlow, and some code for simple reward models.

.. _PhD thesis:
    http://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf

.. note::
    Our current implementation of MCE IRL uses Jax, which is not bundled with the
    default ``pip`` installation of ``imitation`` because Jax is incompatible with
    Windows. You can install the Jax dependencies used by MCE IRL via
    ``pip install imitation[jax]``.

    We are considering porting MCE IRL from Jax to PyTorch in a future release.
"""

import abc
import logging

import numpy as np
import scipy

try:
    # pytype: disable=import-error
    import jax
    import jax.experimental.stax as jstax
    import jax.numpy as jnp
    import jax.random as jrandom

    # pytype: enable=import-error
except ImportError as e:  # pragma: no cover
    msg = (
        f"Failed to import module {__name__} because Jax dependency is not installed. "
        "See module docstring for more information on installation and OS "
        "compatibility."
    )
    # ImportWarning is more appropriate than UserWarning here, but ImportWarning
    # has been ignored by default since Python 3.7:
    # https://docs.python.org/3/library/devmode.html#effects-of-the-python-development-mode
    raise ImportError(msg) from e


def mce_partition_fh(env, *, R=None):
    r"""Performs the soft Bellman backup for a finite-horizon, undiscounted MDP.

    Calculates V^{soft}, Q^{soft}, and pi using recurrences (9.1), (9.2), and
    (9.3) from Ziebart (2010).

    Args:
        env (ModelBasedEnv): a tabular, known-dynamics MDP.
        R (None or np.array): a reward matrix. Defaults to env.reward_matrix.

    Returns:
        (V, Q, \pi) corresponding to the soft values, Q-values and MCE policy.
        V is a 2d array, indexed V[t,s]. Q is a 3d array, indexed Q[t,s,a].
        \pi is a 3d array, indexed \pi[t,s,a].
    """

    # shorthand
    horizon = env.horizon
    n_states = env.n_states
    n_actions = env.n_actions
    T = env.transition_matrix
    if R is None:
        R = env.reward_matrix

    # Initialization
    # indexed as V[t,s]
    V = np.full((horizon, n_states), -np.inf)
    # indexed as Q[t,s,a]
    Q = np.zeros((horizon, n_states, n_actions))
    broad_R = R[:, None]

    # Base case: final timestep
    # final Q(s,a) is just reward
    Q[horizon - 1, :, :] = broad_R
    # V(s) is always normalising constant
    V[horizon - 1, :] = scipy.special.logsumexp(Q[horizon - 1, :, :], axis=1)

    # Recursive case
    for t in reversed(range(horizon - 1)):
        next_values_s_a = T @ V[t + 1, :]
        Q[t, :, :] = broad_R + next_values_s_a
        V[t, :] = scipy.special.logsumexp(Q[t, :, :], axis=1)

    pi = np.exp(Q - V[:, :, None])

    return V, Q, pi


def mce_occupancy_measures(env, *, R=None, pi=None):
    """Calculate state visitation frequency Ds for each state s under a given policy pi.

    You can get pi from `mce_partition_fh`.

    Args:
        env (ModelBasedEnv): a tabular MDP.
        R (None or np.ndarray): reward matrix. Defaults is env.reward_matrix.
        pi (None or np.ndarray): policy to simulate. Defaults to soft-optimal
            policy w.r.t reward matrix.

    Returns:
        Tuple of D (ndarray) and Dt (ndarray). D is an :math:`|S|`-dimensional
        vector recording the expected number of times each state is visited.
        Dt is a :math:`T*|S|`-dimensional vector recording the probability of
        being in a given state at a given timestep.
    """

    # shorthand
    horizon = env.horizon
    n_states = env.n_states
    n_actions = env.n_actions
    T = env.transition_matrix
    if R is None:
        R = env.reward_matrix
    if pi is None:
        _, _, pi = mce_partition_fh(env, R=R)

    D = np.zeros((horizon, n_states))
    D[0, :] = env.initial_state_dist
    for t in range(1, horizon):
        for a in range(n_actions):
            E = D[t - 1] * pi[t - 1, :, a]
            D[t, :] += E @ T[:, a, :]

    return D, D.sum(axis=0)


def mce_irl(
    env,
    optimiser_tuple,
    rmodel,
    demo_state_om,
    linf_eps=1e-3,
    grad_l2_eps=1e-4,
    print_interval=100,
):
    r"""Tabular MCE IRL.

    Args:
        env (ModelBasedEnv): a tabular MDP.
        optimiser_tuple (tuple): a tuple of `(optim_init_fn, optim_update_fn,
            get_params_fn)` produced by a Jax optimiser.
        rmodel (RewardModel): a reward function to be optimised.
        demo_state_om (np.ndarray): matrix representing state occupancy measure
            for demonstrator.
        linf_eps (float): optimisation terminates if the $l_{\infty}$ distance
            between the demonstrator's state occupancy measure and the state
            occupancy measure for the current reward falls below this value.
        grad_l2_eps (float): optimisation also terminates if the $\ell_2$ norm
            of the MCE IRL gradient falls below this value.
        print_interval (int or None): how often to log current loss stats
            (using `logging`). None to disable.

    Returns:
        (np.ndarray, np.ndarray): tuple of final parameters found by optimiser
        and state occupancy measure for the final reward function. Note that
        rmodel will also be updated with the latest parameters."""

    obs_mat = env.observation_matrix
    # l_\infty distance between demonstrator occupancy measure (OM) and OM for
    # soft-optimal policy w.r.t current reward (initially set to this value to
    # prevent termination)
    linf_delta = linf_eps + 1
    # norm of the MCE IRL gradient (also set to this value to prevent
    # termination)
    grad_norm = grad_l2_eps + 1
    # number of optimisation steps taken
    t = 0
    assert demo_state_om.shape == (len(obs_mat),)
    opt_init, opt_update, opt_get_params = optimiser_tuple
    rew_params = rmodel.get_params()
    opt_state = opt_init(rew_params)

    while linf_delta > linf_eps and grad_norm > grad_l2_eps:
        # get reward predicted for each state by current model, & compute
        # expected # of times each state is visited by soft-optimal policy
        # w.r.t that reward function
        predicted_r, out_grads = rmodel.out_grads(obs_mat)
        _, visitations = mce_occupancy_measures(env, R=predicted_r)
        # gradient of partition function w.r.t parameters; equiv to expectation
        # over states drawn from current imitation distribution of the gradient
        # of the reward function w.r.t its params
        pol_grad = np.sum(visitations[:, None] * out_grads, axis=0)
        # gradient of reward function w.r.t parameters, with expectation taken
        # over states
        expert_grad = np.sum(demo_state_om[:, None] * out_grads, axis=0)
        grad = pol_grad - expert_grad

        # these are just for termination conditions & debug logging
        grad_norm = np.linalg.norm(grad)
        linf_delta = np.max(np.abs(demo_state_om - visitations))
        if print_interval is not None and 0 == (t % print_interval):
            logging.info(
                "Occupancy measure error@iter % 3d: %f (||params||=%f, "
                "||grad||=%f, ||E[dr/dw]||=%f)"
                % (
                    t,
                    linf_delta,
                    np.linalg.norm(rew_params),
                    np.linalg.norm(grad),
                    np.linalg.norm(pol_grad),
                )
            )

        # take a single optimiser step
        opt_state = opt_update(t, grad, opt_state)
        rew_params = opt_get_params(opt_state)
        rmodel.set_params(rew_params)
        t += 1

    return rew_params, visitations


# ############################### #
# ####### REWARD MODELS ######### #
# ############################### #


class RewardModel(abc.ABC):
    """Abstract model for reward functions (linear, MLPs, nearest-neighbour, etc.)"""

    @abc.abstractmethod
    def out(self, inputs):
        """Get rewards for a batch of observations.

        Args:
            inputs (np.ndarray): 2D matrix of observations, with first axis
                most likely indexing over state & second indexing over elements
                of observations themselves.

        Returns:
            np.ndarray of rewards (just a 1D vector with one element for each
            supplied observation).
        """

    @abc.abstractmethod
    def grads(self, inputs):
        """Gradients of reward with respect to a batch of input observations.

        Args:
            inputs (np.ndarray): 2D matrix of observations, like .out().

        Returns:
            np.ndarray of gradients *with respect to each input separately*.
            e.g if the model has a W-dimensional parameter vector, and there
            are O observation passed in, then the return value will be an O*W
            matrix of gradients.
        """

    def out_grads(self, inputs):
        """Combination method to do forward-prop AND back-prop. This is trivial for
        linear models, but might provide some cost saving for deep ones.

        Args:
            inputs (np.ndarray): 2D matrix of observations, like .out().

        Returns:
            (np.ndarray, np.ndarray), where first array is equivalent to return
            value of .out() and second array is equivalent to return value of
            .grads().
        """
        return self.out(inputs), self.grads(inputs)

    @abc.abstractmethod
    def set_params(self, params):
        """Set a new parameter vector for the model (from flat Numpy array).

        Args:
            params (np.ndarray): 1D parameter vector for the model.
        """

    @abc.abstractmethod
    def get_params(self):
        """Get current parameter vector from model (as flat Numpy array).

        Args: empty.

        Returns:
            np.ndarray: 1D parameter vector for the model.
        """


class LinearRewardModel(RewardModel):
    """Linear reward model (without bias)."""

    def __init__(self, obs_dim, *, seed=None):
        """Construct linear reward model for `obs_dim`-dimensional observation space.

        Initial values are generated from given seed (int or None).

        Args:
            obs_dim (int): dimensionality of observation space.
            seed (int or None): random seed for generating initial params. If
                None, seed will be chosen arbitrarily
        """
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        self._weights = rng.randn(obs_dim)

    def out(self, inputs):
        assert inputs.shape[1:] == self._weights.shape
        return inputs @ self._weights

    def grads(self, inputs):
        assert inputs.shape[1:] == self._weights.shape
        return inputs

    def set_params(self, params):
        assert params.shape == self._weights.shape
        self._weights = params

    def get_params(self):
        return self._weights


class JaxRewardModel(RewardModel, abc.ABC):
    """Wrapper for arbitrary Jax-based reward models.

    Useful for neural nets.
    """

    def __init__(self, obs_dim, *, seed=None):
        """Internal setup for Jax-based reward models.

        Initialises reward model using given seed & input size (`obs_dim`).

        Args:
            obs_dim (int): dimensionality of observation space.
            seed (int or None): random seed for generating initial params. If
                None, seed will be chosen arbitrarily, as in
                LinearRewardModel.
        """
        # TODO: apply jax.jit() to everything in sight
        net_init, self._net_apply = self.make_stax_model()
        if seed is None:
            # oh well
            seed = np.random.randint((1 << 63) - 1)
        rng = jrandom.PRNGKey(seed)
        out_shape, self._net_params = net_init(rng, (-1, obs_dim))
        self._net_grads = jax.grad(self._net_apply)
        # output shape should just be batch dim, nothing else
        assert out_shape == (-1,), "got a weird output shape %s" % (out_shape,)

    @abc.abstractmethod
    def make_stax_model(self):
        """Build the stax model that this thing is meant to optimise.

        Should return (net_init, net_apply) pair, just like Stax modules.

        Returns:
            tuple of net_init(rng, input_shape) function to initialise the
            network, and net_apply(params, inputs) to do forward prop on the
            network.
        """

    def _flatten(self, matrix_tups):
        """Flatten everything and concatenate it together."""
        out_vecs = [v.flatten() for t in matrix_tups for v in t]
        return jnp.concatenate(out_vecs)

    def _flatten_batch(self, matrix_tups):
        """Flatten all except leading dim & concatenate results together in channel dim.

        (Channel dim is whatever the dim after the leading dim is)."""
        out_vecs = []
        for t in matrix_tups:
            for v in t:
                new_shape = (v.shape[0],)
                if len(v.shape) > 1:
                    new_shape = new_shape + (np.prod(v.shape[1:]),)
                out_vecs.append(v.reshape(new_shape))
        return jnp.concatenate(out_vecs, axis=1)

    def out(self, inputs):
        return np.asarray(self._net_apply(self._net_params, inputs))

    def grads(self, inputs):
        in_grad_partial = jax.partial(self._net_grads, self._net_params)
        grad_vmap = jax.vmap(in_grad_partial)
        rich_grads = grad_vmap(inputs)
        flat_grads = np.asarray(self._flatten_batch(rich_grads))
        assert flat_grads.ndim == 2 and flat_grads.shape[0] == inputs.shape[0]
        return flat_grads

    def set_params(self, params):
        # have to reconstitute appropriately-shaped weights from 1D param vec
        # shit this is going to be annoying
        idx_acc = 0
        new_params = []
        for t in self._net_params:
            new_t = []
            for v in t:
                new_idx_acc = idx_acc + v.size
                new_v = params[idx_acc:new_idx_acc].reshape(v.shape)
                # this seems to cast it to Jax DeviceArray appropriately;
                # surely there's better way, though?
                new_v = 0.0 * v + new_v
                new_t.append(new_v)
                idx_acc = new_idx_acc
            new_params.append(new_t)
        self._net_params = new_params

    def get_params(self):
        return self._flatten(self._net_params)


class MLPRewardModel(JaxRewardModel):
    """Simple MLP-based reward function with Jax/Stax."""

    def __init__(self, obs_dim, hiddens, activation="Tanh", **kwargs):
        """Construct an MLP-based reward function.

        Args:
            obs_dim (int): dimensionality of observation space.
            hiddens ([int]): size of hidden layers.
            activation (str): name of activation (Tanh, Relu, Softplus
                supported).
            **kwargs: extra keyword arguments to be passed to
                JaxRewardModel.__init__().
        """
        assert activation in ["Tanh", "Relu", "Softplus"], (
            "probably can't handle activation '%s'" % activation
        )
        self._hiddens = hiddens
        self._activation = activation
        super().__init__(obs_dim, **kwargs)

    def make_stax_model(self):
        act = getattr(jstax, self._activation)
        layers = []
        for h in self._hiddens:
            layers.extend([jstax.Dense(h), act])
        layers.extend([jstax.Dense(1), _StaxSqueeze()])
        return jstax.serial(*layers)


def _StaxSqueeze(axis=-1):
    """Stax layer that collapses a single axis that has dimension 1.

    Only used in MLPRewardModel.
    """

    def init_fun(rng, input_shape):
        ax = axis
        if ax < 0:
            ax = len(input_shape) + ax
        assert ax < len(input_shape), "invalid axis %d for %d-dimensional tensor" % (
            axis,
            len(input_shape),
        )
        assert input_shape[ax] == 1, "axis %d is %d, not 1" % (axis, input_shape[ax])
        output_shape = input_shape[:ax] + input_shape[ax + 1 :]
        return output_shape, ()

    def apply_fun(params, inputs, **kwargs):
        return jnp.squeeze(inputs, axis=axis)

    return init_fun, apply_fun
