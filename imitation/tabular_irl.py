"""Finite-horizon tabular Maximum Causal Entropy IRL.

Follows the description in Brian Ziebart's PhD thesis (2010), see
chapters 9 and 10 of:
    http://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf

Uses NumPy-based optimizer Jax, so the code can be run without
PyTorch/TensorFlow, and some code for simple reward models."""

import abc

import jax
import jax.experimental.stax as jstax
import jax.numpy as jnp
import jax.random as jrandom
import numpy as np
import scipy
import tensorflow as tf


def mce_partition_fh(env, *, R=None):
    """Performs the soft Bellman backup for a finite-horizon, undiscounted MDP.

    Calculates V^{soft}, Q^{soft}, and pi using recurrences (9.1), (9.2), and
    (9.3) from Ziebart (2010).

    Args:
        env (ModelBasedEnv): a tabular, known-dynamics MDP.
        R (None or np.array): a reward matrix. Defaults to env.reward_matrix.

    Returns:
        (V, Q, pi) corresponding to the soft values, Q-values and MCE policy.
        V is a 2d array, indexed V[t,s]. Q is a 3d array, indexed Q[t,s,a].
        pi is a 3d array, indexed pi[t,s,a].
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
    for t in range(horizon - 1)[::-1]:
        next_values_s_a = T @ V[t + 1, :]
        Q[t, :, :] = broad_R + next_values_s_a
        V[t, :] = scipy.special.logsumexp(Q[t, :, :], axis=1)

    pi = np.exp(Q - V[:, :, None])

    return V, Q, pi


def mce_occupancy_measures(env, *, R=None, pi=None):
    """Calculate state visitation frequency Ds for each state s under a given
    policy pi. You can get pi from `mce_partition_fh`.

    Args:
        env (ModelBasedEnv): a tabular MDP.
        R (None or np.ndarray): reward matrix. Defaults is env.reward_matrix.
        pi (None or np.ndarray): policy to simulate. Defaults to soft-optimal
            policy w.r.t reward matrix.

    Returns:
        Tuple of D (ndarray) and Dt (ndarray). D is an |S|-dimensional vector
        recording the expected number of times each state is visited. Dt is a
        T*|S|-dimensional vector recording the probability of being in a given
        state at a given timestep."""

    # shorthand
    horizon = env.horizon
    n_states = env.n_states
    n_actions = env.n_actions
    T = env.transition_matrix
    if R is None:
        R = env.reward_matrix
    if pi is None:
        _, _, pi = mce_partition_fh(env, R=R)

    # we always start in s0, WLOG (for other distributions, just make all
    # actions in s0 take you to random state)
    init_states = np.zeros((n_states))
    init_states[0] = 1

    D = np.zeros((horizon, n_states))
    D[0, :] = init_states
    for t in range(1, horizon):
        for a in range(n_actions):
            E = D[t - 1] * pi[t - 1, :, a]
            D[t, :] += E @ T[:, a, :]

    return D, D.sum(axis=0)


def maxent_irl(
        env,
        optimiser,
        rmodel,
        demo_state_om,
        linf_eps=1e-3,
        grad_l2_eps=1e-4,
        print_interval=100):
    r"""Tabular MCE IRL.

    Args:
        env (ModelBasedEnv): a tabular MDP.
        rmodel (RewardModel): a reward function to be optimised.
        demo_state_om (np.ndarray): matrix representing state occupancy measure
            for demonstrator.
        linf_eps (float): optimisation terminates if the $l_\infty$ distance
            between the demonstrator's state occupancy measure and the state
            occupancy measure for the current reward falls below this value.
        grad_l2_eps (float): optimisation also terminates if the $\ell_2$ norm
            of the MCE IRL gradient falls below this value.
        print_interval (int or None): how often to log current loss stats
            (using tf.logging). None to disable.

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
    assert demo_state_om.shape == (len(obs_mat), )
    rew_params = optimiser.current_params
    rmodel.set_params(rew_params)

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
            tf.logging.info(
                'Occupancy measure error@iter % 3d: %f (||params||=%f, '
                '||grad||=%f, ||E[dr/dw]||=%f)' %
                (t, linf_delta, np.linalg.norm(rew_params),
                 np.linalg.norm(grad), np.linalg.norm(pol_grad)))

        # take a single optimiser step
        optimiser.step(grad)
        rew_params = optimiser.current_params
        rmodel.set_params(rew_params)
        t += 1

    return optimiser.current_params, visitations


# ############################### #
# ####### REWARD MODELS ######### #
# ############################### #


class RewardModel(abc.ABC):
    """Abstract model for reward functions (which might be linear, MLPs,
    nearest-neighbour, etc.)"""

    @abc.abstractmethod
    def out(self, inputs):
        """Get rewards for a batch of observations."""

    @abc.abstractmethod
    def grads(self, inputs):
        """Gradients of reward with respect to a batch of input observations."""

    def out_grads(self, inputs):
        """Combination method to do forward-prop AND back-prop (trivial for
        linear models, maybe some cost saving for deep model)."""
        return self.out(inputs), self.grads(inputs)

    @abc.abstractmethod
    def set_params(self, params):
        """Set a new parameter vector for the model (from flat Numpy array)."""

    @abc.abstractmethod
    def get_params(self):
        """Get current parameter vector from model (as flat Numpy array)."""


class LinearRewardModel(RewardModel):
    """Linear reward model (without bias)."""

    def __init__(self, obs_dim, *, seed=None):
        """Construct linear reward model for `obs_dim`-dimensional observation
        space. Initial values are generated from given seed (int or None)."""
        if seed is not None:
            rng = np.random.RandomState(seed)
        else:
            rng = np.random
        self._weights = rng.randn(obs_dim, )

    def out(self, inputs):
        """Get rewards for a batch of observations."""
        assert inputs.shape[1:] == self._weights.shape
        return inputs @ self._weights

    def grads(self, inputs):
        """Individual gradient of reward with respect to each element in a
        batch of input observations."""
        assert inputs.shape[1:] == self._weights.shape
        return inputs

    def set_params(self, params):
        """Set a new parameter vector for the model (from flat Numpy array)."""
        assert params.shape == self._weights.shape
        self._weights = params

    def get_params(self):
        """Get current parameter vector from model (as flat Numpy array)."""
        return self._weights


class JaxRewardModel(RewardModel, abc.ABC):
    """Wrapper for arbitrary Jax-based reward models. Useful for neural
    nets."""

    def __init__(self, obs_dim, *, seed=None):
        """Internal setup for Jax-based reward models. Initialises reward model
        using given seed & input size (`obs_dim`)."""
        # TODO: apply jax.jit() to everything in sight
        net_init, self._net_apply = self.make_stax_model()
        if seed is None:
            # oh well
            seed = np.random.randint((1 << 63) - 1)
        rng = jrandom.PRNGKey(seed)
        out_shape, self._net_params = net_init(rng, (-1, obs_dim))
        self._net_grads = jax.grad(self._net_apply)
        # output shape should just be batch dim, nothing else
        assert out_shape == (-1,), \
            "got a weird output shape %s" % (out_shape,)

    @abc.abstractmethod
    def make_stax_model(self):
        """Build the stax model that this thing is meant to optimise. Should
        return (net_init, net_apply) pair, just like Stax modules."""

    def _flatten(self, matrix_tups):
        """Flatten everything and concatenate it together."""
        out_vecs = [v.flatten() for t in matrix_tups for v in t]
        return jnp.concatenate(out_vecs)

    def _flatten_batch(self, matrix_tups):
        """Flatten all except leading dim & concatenate results together in
        channel dim (i.e whatever the dim after the leading dim is)."""
        out_vecs = []
        for t in matrix_tups:
            for v in t:
                new_shape = (v.shape[0], )
                if len(v.shape) > 1:
                    new_shape = new_shape + (np.prod(v.shape[1:]), )
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

    def __init__(self, obs_dim, hiddens, activation='Tanh', **kwargs):
        """Construct an MLP-based reward function with input layer dimension
        `obs_dim`, size of hidden layers given by `hiddens` (`list[int]`), and
        `activation` nonlinearity (can be `Tanh`, `Relu`, or `Softplus`). Extra
        kwargs are passed to JaxRewardModel.__init__()."""
        assert activation in ['Tanh', 'Relu', 'Softplus'], \
            "probably can't handle activation '%s'" % activation
        self._hiddens = hiddens
        self._activation = activation
        super().__init__(obs_dim, **kwargs)

    def make_stax_model(self):
        act = getattr(jstax, self._activation)
        layers = []
        for h in self._hiddens:
            layers.extend([jstax.Dense(h), act])
        layers.extend([jstax.Dense(1), StaxSqueeze()])
        return jstax.serial(*layers)


def StaxSqueeze(axis=-1):
    """Stax layer that collapses a single axis that has dimension 1."""

    def init_fun(rng, input_shape):
        ax = axis
        if ax < 0:
            ax = len(input_shape) + ax
        assert ax < len(input_shape), \
            "invalid axis %d for %d-dimensional tensor" \
            % (axis, len(input_shape))
        assert input_shape[ax] == 1, "axis %d is %d, not 1" \
            % (axis, input_shape[ax])
        output_shape = input_shape[:ax] + input_shape[ax + 1:]
        return output_shape, ()

    def apply_fun(params, inputs, **kwargs):
        return jnp.squeeze(inputs, axis=axis)

    return init_fun, apply_fun


# ############################### #
# ######### OPTIMISERS ########## #
# ############################### #


class Schedule(abc.ABC):
    """Base class for learning rate schedules."""

    @abc.abstractmethod
    def __iter__(self):
        """Return an iterable of step sizes."""


class ConstantSchedule(Schedule):
    """Constant step size schedule."""

    def __init__(self, lr):
        self.lr = lr

    def __iter__(self):
        while True:
            yield self.lr


class SqrtTSchedule(Schedule):
    """1/sqrt(t) step size schedule."""

    def __init__(self, init_lr):
        self.init_lr = init_lr

    def __iter__(self):
        t = 1
        while True:
            yield self.init_lr / np.sqrt(t)
            t += 1


def get_schedule(lr_or_schedule):
    """Turn a constant float/int or an actual Schedule into a canonical
    Schedule instance."""
    if isinstance(lr_or_schedule, Schedule):
        return lr_or_schedule
    if isinstance(lr_or_schedule, (float, int)):
        return ConstantSchedule(lr_or_schedule)
    raise TypeError("No idea how to make schedule out of '%s'" %
                    lr_or_schedule)


class Optimiser(abc.ABC):
    """Abstract base class for optimisers like Nesterov, Adam, etc."""

    @abc.abstractmethod
    def step(self, grad):
        """Take a step using the supplied gradient vector."""

    @property
    @abc.abstractmethod
    def current_params(self):
        """Return the parameters corresponding to the current iterate."""


class AMSGrad(Optimiser):
    """Kind-of-fixed version of Adam optimiser, as described in
    https://openreview.net/pdf?id=ryQu7f-RZ. This should roughly correspond to
    a diagonal approximation to natural gradient, just as Adam does, but
    without the pesky non-convergence issues."""

    def __init__(self,
                 rmodel,
                 alpha_sched=1e-3,
                 beta1=0.9,
                 beta2=0.99,
                 eps=1e-8):
        # x is initial parameter vector; alpha is step size; beta1 & beta2 are
        # as defined in AMSGrad paper; eps is added to sqrt(vhat) during
        # calculation of next iterate to ensure division does not overflow.
        init_params = rmodel.get_params()
        param_size, = init_params.shape
        # first moment estimate
        self.m = np.zeros((param_size, ))
        # second moment estimate
        self.v = np.zeros((param_size, ))
        # max second moment
        self.vhat = np.zeros((param_size, ))
        # parameter estimate
        self.x = init_params
        # step sizes etc.
        self.alpha_schedule = iter(get_schedule(alpha_sched))
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

    def step(self, grad):
        alpha = next(self.alpha_schedule)
        self.m = self.beta1 * self.m + (1 - self.beta1) * grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
        self.vhat = np.maximum(self.vhat, self.v)
        # 1e-5 for numerical stability
        denom = np.sqrt(self.vhat) + self.eps
        self.x = self.x - alpha * self.m / denom
        return self.x

    @property
    def current_params(self):
        return self.x


class SGD(Optimiser):
    """Standard gradient method."""

    def __init__(self, rmodel, alpha_sched=1e-3):
        init_params = rmodel.get_params()
        self.x = init_params
        self.alpha_schedule = iter(get_schedule(alpha_sched))
        self.cnt = 1

    def step(self, grad):
        alpha = next(self.alpha_schedule)
        self.x = self.x - alpha * grad
        return self.x

    @property
    def current_params(self):
        return self.x
