"""Finite-horizon tabular Maximum Causal Entropy IRL.

Follows the description in chapters 9 and 10 of Brian Ziebart's `PhD thesis`_.

.. _PhD thesis:
    http://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf
"""

from typing import Any, Iterable, Mapping, Optional, Tuple, Type, Union

import numpy as np
import scipy.special
import torch as th

from imitation.algorithms import base
from imitation.data import rollout, types
from imitation.envs import resettable_env
from imitation.rewards import reward_nets
from imitation.util import logger as imit_logger
from imitation.util import util


def mce_partition_fh(
    env: resettable_env.TabularModelEnv,
    *,
    reward: Optional[np.ndarray] = None,
    discount: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    r"""Performs the soft Bellman backup for a finite-horizon MDP.

    Calculates V^{soft}, Q^{soft}, and pi using recurrences (9.1), (9.2), and
    (9.3) from Ziebart (2010).

    Args:
        env: a tabular, known-dynamics MDP.
        reward: a reward matrix. Defaults to env.reward_matrix.
        discount: discount rate.

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
    if reward is None:
        reward = env.reward_matrix

    # Initialization
    # indexed as V[t,s]
    V = np.full((horizon, n_states), -np.inf)
    # indexed as Q[t,s,a]
    Q = np.zeros((horizon, n_states, n_actions))
    broad_R = reward[:, None]

    # Base case: final timestep
    # final Q(s,a) is just reward
    Q[horizon - 1, :, :] = broad_R
    # V(s) is always normalising constant
    V[horizon - 1, :] = scipy.special.logsumexp(Q[horizon - 1, :, :], axis=1)

    # Recursive case
    for t in reversed(range(horizon - 1)):
        next_values_s_a = T @ V[t + 1, :]
        Q[t, :, :] = broad_R + discount * next_values_s_a
        V[t, :] = scipy.special.logsumexp(Q[t, :, :], axis=1)

    pi = np.exp(Q - V[:, :, None])

    return V, Q, pi


def mce_occupancy_measures(
    env: resettable_env.TabularModelEnv,
    *,
    reward: Optional[np.ndarray] = None,
    pi: Optional[np.ndarray] = None,
    discount: float = 1.0,
) -> Tuple[np.ndarray, np.ndarray]:
    """Calculate state visitation frequency Ds for each state s under a given policy pi.

    You can get pi from `mce_partition_fh`.

    Args:
        env: a tabular MDP.
        reward: reward matrix. Defaults is env.reward_matrix.
        pi: policy to simulate. Defaults to soft-optimal policy w.r.t reward
            matrix.
        discount: rate to discount the cumulative occupancy measure D.

    Returns:
        Tuple of D (ndarray) and Dt (ndarray). D is an :math:`|S|`-dimensional vector
        recording the expected discounted number of times each state is visited. Dt is
        a :math:`T*|S|`-dimensional vector recording the probability of being in a given
        state at a given timestep.
    """
    # shorthand
    horizon = env.horizon
    n_states = env.n_states
    n_actions = env.n_actions
    T = env.transition_matrix
    if reward is None:
        reward = env.reward_matrix
    if pi is None:
        _, _, pi = mce_partition_fh(env, reward=reward)

    D = np.zeros((horizon, n_states))
    D[0, :] = env.initial_state_dist
    for t in range(1, horizon):
        for a in range(n_actions):
            E = D[t - 1] * pi[t - 1, :, a]
            D[t, :] += E @ T[:, a, :]

    Dcum = rollout.discounted_sum(D, discount)
    assert isinstance(Dcum, np.ndarray)
    return D, Dcum


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


MCEDemonstrations = Union[np.ndarray, base.AnyTransitions]


class MCEIRL(base.DemonstrationAlgorithm[types.TransitionsMinimal]):
    """Tabular MCE IRL."""

    demo_state_om: Optional[np.ndarray]

    def __init__(
        self,
        demonstrations: Optional[MCEDemonstrations],
        env: resettable_env.TabularModelEnv,
        reward_net: Optional[reward_nets.RewardNet] = None,
        optimizer_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        discount: float = 1,
        linf_eps: float = 1e-3,
        grad_l2_eps: float = 1e-4,
        # TODO(adam): do we need log_interval or can just use record_mean...?
        log_interval: Optional[int] = 100,
        *,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ):
        r"""Creates MCE IRL.

        Args:
            demonstrations: Demonstrations from an expert (optional). Can be a sequence
                of trajectories, or transitions, an iterable over mappings that
                represent a batch of transitions, or a state occupancy measure.
                The demonstrations must have observations one-hot coded unless
                demonstrations is a state-occupancy measure.
            env: a tabular MDP.
            reward_net: a neural network that computes rewards for the supplied
                observations.
            optimizer_cls: optimiser to use for supervised training.
            optimizer_kwargs: keyword arguments for optimiser construction.
            discount: the discount factor to use when computing occupancy measure.
                If not 1.0 (undiscounted), then `demonstrations` must either be
                a (discounted) state-occupancy measure, or trajectories. Transitions
                are *not allowed* as we cannot discount them appropriately without
                knowing the timestep they were drawn from.
            linf_eps: optimisation terminates if the $l_{\infty}$ distance between
                the demonstrator's state occupancy measure and the state occupancy
                measure for the current reward falls below this value.
            grad_l2_eps: optimisation also terminates if the $\ell_2$ norm of the
                MCE IRL gradient falls below this value.
            log_interval: how often to log current loss stats (using `logging`).
                None to disable.
            custom_logger: Where to log to; if None (default), creates a new logger.
        """
        self.discount = discount
        self.env = env
        self.demo_state_om = None
        super().__init__(
            demonstrations=demonstrations,
            custom_logger=custom_logger,
        )

        if reward_net is None:
            reward_net = reward_nets.BasicRewardNet(
                env.observation_space,
                env.action_space,
                use_action=False,
                use_next_state=False,
                use_done=False,
                hid_sizes=[],
            )
        self.reward_net = reward_net
        optimizer_kwargs = optimizer_kwargs or {"lr": 1e-2}
        self.optimizer = optimizer_cls(reward_net.parameters(), **optimizer_kwargs)

        self.linf_eps = linf_eps
        self.grad_l2_eps = grad_l2_eps
        self.log_interval = log_interval

    def _set_demo_from_obs(self, obses: np.ndarray) -> None:
        if self.discount != 1.0:
            raise ValueError(
                "Cannot compute discounted OM from timeless Transitions.",
            )
        for obs in obses:
            if isinstance(obs, th.Tensor):
                obs = obs.numpy()
            self.demo_state_om[obs.astype(bool)] += 1.0

    def set_demonstrations(self, demonstrations: MCEDemonstrations) -> None:
        if isinstance(demonstrations, np.ndarray):
            # Demonstrations are an occupancy measure
            assert demonstrations.ndim == 1
            self.demo_state_om = demonstrations
        else:
            # Demonstrations are either trajectories or transitions;
            # we must compute occupancy measure from this.
            self.demo_state_om = np.zeros((self.env.n_states,))

            if isinstance(demonstrations, Iterable):
                first_item = next(iter(demonstrations))
                if isinstance(first_item, types.Trajectory):
                    # Demonstrations are trajectories.
                    for traj in demonstrations:
                        # TODO(adam): vectorize?
                        cum_discount = 1
                        for obs in traj.obs:
                            self.demo_state_om[obs.astype(bool)] += cum_discount
                            cum_discount *= self.discount
                else:
                    # Demonstrations are a Torch DataLoader or other Mapping iterable
                    for batch in demonstrations:
                        self._set_demo_from_obs(batch["obs"])

            elif isinstance(demonstrations, types.TransitionsMinimal):
                self._set_demo_from_obs(demonstrations.obs)
            else:
                raise TypeError(
                    f"Unsupported demonstration type {type(demonstrations)}",
                )

            self.demo_state_om /= self.demo_state_om.sum()  # normalize

    def train(self, max_iter: int = 1000) -> np.ndarray:
        """Runs MCE IRL.

        Args:
            max_iter: The maximum number of iterations to train for. May terminate
                earlier if `self.linf_eps` or `self.grad_l2_eps` thresholds are reached.

        Returns:
            State occupancy measure for the final reward function. `self.reward_net`
            and `self.optimizer` will be updated in-place during optimisation.
        """
        # use the same device and dtype as the rmodel parameters
        obs_mat = self.env.observation_matrix
        dtype = self.reward_net.dtype
        device = self.reward_net.device
        torch_obs_mat = th.as_tensor(
            obs_mat,
            dtype=dtype,
            device=device,
        )
        assert self.demo_state_om.shape == (len(obs_mat),)

        for t in range(max_iter):
            self.optimizer.zero_grad()

            # get reward predicted for each state by current model, & compute
            # expected # of times each state is visited by soft-optimal policy
            # w.r.t that reward function
            # TODO(adam): support not just state-only reward?
            predicted_r = squeeze_r(self.reward_net(torch_obs_mat, None, None, None))
            assert predicted_r.shape == (obs_mat.shape[0],)
            predicted_r_np = predicted_r.detach().cpu().numpy()
            _, visitations = mce_occupancy_measures(
                self.env,
                reward=predicted_r_np,
                discount=self.discount,
            )

            # Forward/back/step (grads are zeroed at the top).
            # weights_th(s) = \pi(s) - D(s)
            weights_th = th.as_tensor(
                visitations - self.demo_state_om,
                dtype=dtype,
                device=device,
            )
            # The "loss" is then:
            #   E_\pi[r_\theta(S)] - E_D[r_\theta(S)]
            loss = th.dot(weights_th, predicted_r)
            # This gives the required gradient:
            #   E_\pi[\nabla r_\theta(S)] - E_D[\nabla r_\theta(S)].
            loss.backward()
            self.optimizer.step()

            # these are just for termination conditions & debug logging
            grad_norm = util.tensor_iter_norm(
                p.grad for p in self.reward_net.parameters()
            ).item()
            linf_delta = np.max(np.abs(self.demo_state_om - visitations))

            if self.log_interval is not None and 0 == (t % self.log_interval):
                weight_norm = util.tensor_iter_norm(self.reward_net.parameters()).item()
                self.logger.record("iteration", t)
                self.logger.record("linf_delta", linf_delta)
                self.logger.record("weight_norm", weight_norm)
                self.logger.record("grad_norm", grad_norm)
                self.logger.dump(t)

            if linf_delta <= self.linf_eps or grad_norm <= self.grad_l2_eps:
                break

        return visitations
