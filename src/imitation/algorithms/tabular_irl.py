"""Finite-horizon tabular Maximum Causal Entropy IRL.

Follows the description in chapters 9 and 10 of Brian Ziebart's `PhD thesis`_.

.. _PhD thesis:
    http://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf
"""

import logging
from typing import Callable, Iterable, Optional, Tuple

import numpy as np
import scipy.special
import torch as th
import torch.nn as nn
from torch.optim.optimizer import Optimizer

from imitation.envs.resettable_env import TabularModelEnv
from imitation.util.util import tensor_iter_norm


def mce_partition_fh(
    env: TabularModelEnv,
    *,
    R: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Performs the soft Bellman backup for a finite-horizon, undiscounted MDP.

    Calculates V^{soft}, Q^{soft}, and pi using recurrences (9.1), (9.2), and
    (9.3) from Ziebart (2010).

    Args:
        env: a tabular, known-dynamics MDP.
        R: a reward matrix. Defaults to env.reward_matrix.

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


def mce_occupancy_measures(
    env: TabularModelEnv,
    *,
    R: Optional[np.ndarray] = None,
    pi: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate state visitation frequency Ds for each state s under a given policy pi.

    You can get pi from `mce_partition_fh`.

    Args:
        env: a tabular MDP.
        R: reward matrix. Defaults is env.reward_matrix.
        pi: policy to simulate. Defaults to soft-optimal policy w.r.t reward
            matrix.

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


def squeeze_r(r_output: th.Tensor) -> th.Tensor:
    """Squeeze a reward output tensor down to one dimension, if necessary.

    Args:
         r_output (th.Tensor): output of reward model. Can be either 1D
            ([n_states]) or 2D ([n_states, 1]).

    Returns:
         squeezed reward of shape [n_states].
    """
    if r_output.ndim == 2:
        return th.squeeze(r_output, 1)
    assert r_output.ndim == 1
    return r_output


def mce_irl(
    env: TabularModelEnv,
    optimizer: Optimizer,
    rmodel: nn.Module,
    demo_state_om: np.ndarray,
    linf_eps: float = 1e-3,
    grad_l2_eps: float = 1e-4,
    print_interval: Optional[int] = 100,
) -> np.ndarray:
    r"""Tabular MCE IRL.

    Args:
        env: a tabular MDP.
        optimizer: an optimizer for `rmodel`.
        rmodel: a neural network that computes rewards for the supplied
            observations.
        demo_state_om: matrix representing state occupancy measure for
            demonstrator.
        linf_eps: optimisation terminates if the $l_{\infty}$ distance between
            the demonstrator's state occupancy measure and the state occupancy
            measure for the current reward falls below this value.
        grad_l2_eps: optimisation also terminates if the $\ell_2$ norm of the
            MCE IRL gradient falls below this value.
        print_interval: how often to log current loss stats (using `logging`).
            None to disable.

    Returns:
        state occupancy measure for the final reward function. Note that `rmodel`
        and `optimizer` will be updated in-place during optimisation.
    """
    # use the same device and dtype as the rmodel parameters
    a_param = next(rmodel.parameters())
    device = a_param.device
    dtype = a_param.dtype

    obs_mat = env.observation_matrix
    torch_obs_mat = th.as_tensor(obs_mat, dtype=dtype, device=device)
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

    while linf_delta > linf_eps and grad_norm > grad_l2_eps:
        optimizer.zero_grad()

        # get reward predicted for each state by current model, & compute
        # expected # of times each state is visited by soft-optimal policy
        # w.r.t that reward function
        predicted_r = squeeze_r(rmodel(torch_obs_mat))
        assert predicted_r.shape == (obs_mat.shape[0],)
        predicted_r_np = predicted_r.detach().cpu().numpy()
        _, visitations = mce_occupancy_measures(env, R=predicted_r_np)

        # Forward/back/step (grads are zeroed at the top).
        # weights_th(s) = \pi(s) - D(s)
        weights_th = th.as_tensor(
            visitations - demo_state_om,
            dtype=dtype,
            device=device,
        )
        # The "loss" is then:
        #   E_\pi[r_\theta(S)] - E_D[r_\theta(S)]
        loss = th.dot(weights_th, predicted_r)
        # This gives the required gradient:
        #   E_\pi[\nabla r_\theta(S)] - E_D[\nabla r_\theta(S)].
        loss.backward()
        optimizer.step()

        # these are just for termination conditions & debug logging
        grad_norm = tensor_iter_norm(p.grad for p in rmodel.parameters()).item()
        linf_delta = np.max(np.abs(demo_state_om - visitations))
        if print_interval is not None and 0 == (t % print_interval):
            weight_norm = tensor_iter_norm(rmodel.parameters()).item()
            logging.info(
                "Occupancy measure error@iter % 3d: %f (||params||=%f, "
                "||grad||=%f)"
                % (
                    t,
                    linf_delta,
                    weight_norm,
                    grad_norm,
                ),
            )

        t += 1

    return visitations


class LinearRewardModel(nn.Module):
    """Simplest possible linear reward model."""

    def __init__(self, obs_dim: int):
        """Construct a linear reward model, with no bias.

        Args:
            obs_dim: size of input observations.
        """
        super().__init__()
        # no bias in this model for the sake of simplicity
        self.lin = nn.Linear(obs_dim, 1, bias=False)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return th.squeeze(self.lin(obs), 1)


class MLPRewardModel(nn.Module):
    """MLP-based reward model."""

    def __init__(
        self,
        obs_dim: int,
        hiddens: Iterable[int],
        activation: Callable[[], nn.Module] = nn.Tanh,
    ):
        """Construct an MLP-based reward model.

        Args:
            obs_dim: size of input observations.
            hiddens: list of zero or more hidden layer sizes.
            activation: constructs an activation to insert between hidden
                layers (e.g. you could supply `nn.ReLU`, `nn.Tanh`, etc.`).
        """
        super().__init__()
        layer_list = []
        prev_h_size = obs_dim
        for h_size in hiddens:
            layer_list.append(nn.Linear(prev_h_size, h_size))
            layer_list.append(activation())
            prev_h_size = h_size
        layer_list.append(nn.Linear(prev_h_size, 1))
        self.layers = nn.Sequential(*layer_list)

    def forward(self, obs: th.Tensor) -> th.Tensor:
        return th.squeeze(self.layers(obs), 1)
