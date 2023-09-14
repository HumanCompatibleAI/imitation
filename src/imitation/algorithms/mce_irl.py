"""Finite-horizon tabular Maximum Causal Entropy IRL.

Follows the description in chapters 9 and 10 of Brian Ziebart's `PhD thesis`_.

.. _PhD thesis:
    http://www.cs.cmu.edu/~bziebart/publications/thesis-bziebart.pdf
"""
import collections
import warnings
from typing import Any, Iterable, List, Mapping, NoReturn, Optional, Tuple, Type, Union

import gym
import numpy as np
import scipy.special
import torch as th
from seals import base_envs
from stable_baselines3.common import policies

from imitation.algorithms import base
from imitation.data import rollout, types
from imitation.rewards import reward_nets
from imitation.util import logger as imit_logger
from imitation.util import networks, util


def mce_partition_fh(
    env: base_envs.TabularModelPOMDP,
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

    Raises:
        ValueError: if ``env.horizon`` is None (infinite horizon).
    """
    # shorthand
    horizon = env.horizon
    if horizon is None:
        raise ValueError("Only finite-horizon environments are supported.")
    n_states = env.state_dim
    n_actions = env.action_dim
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
    env: base_envs.TabularModelPOMDP,
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
        Tuple of ``D`` (ndarray) and ``Dcum`` (ndarray). ``D`` is of shape
        ``(env.horizon, env.n_states)`` and records the probability of being in a
        given state at a given timestep. ``Dcum`` is of shape ``(env.n_states,)``
        and records the expected discounted number of times each state is visited.

    Raises:
        ValueError: if ``env.horizon`` is None (infinite horizon).
    """
    # shorthand
    horizon = env.horizon
    if horizon is None:
        raise ValueError("Only finite-horizon environments are supported.")
    n_states = env.state_dim
    n_actions = env.action_dim
    T = env.transition_matrix
    if reward is None:
        reward = env.reward_matrix
    if pi is None:
        _, _, pi = mce_partition_fh(env, reward=reward)

    D = np.zeros((horizon + 1, n_states))
    D[0, :] = env.initial_state_dist
    for t in range(horizon):
        for a in range(n_actions):
            E = D[t] * pi[t, :, a]
            D[t + 1, :] += E @ T[:, a, :]

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


class TabularPolicy(policies.BasePolicy):
    """A tabular policy. Cannot be trained -- prediction only."""

    pi: np.ndarray
    rng: np.random.Generator

    def __init__(
        self,
        state_space: gym.Space,
        action_space: gym.Space,
        pi: np.ndarray,
        rng: np.random.Generator,
    ) -> None:
        """Builds TabularPolicy.

        Args:
            state_space: The state space of the environment.
            action_space: The action space of the environment.
            pi: A tabular policy. Three-dimensional array, where pi[t,s,a]
                is the probability of taking action a at state s at timestep t.
            rng: Random state, used for sampling when `predict` is called with
                `deterministic=False`.
        """
        assert isinstance(state_space, gym.spaces.Discrete), "state not tabular"
        assert isinstance(action_space, gym.spaces.Discrete), "action not tabular"
        # What we call state space here is observation space in SB3 nomenclature.
        super().__init__(observation_space=state_space, action_space=action_space)
        self.rng = rng
        self.set_pi(pi)

    def set_pi(self, pi: np.ndarray) -> None:
        """Sets tabular policy to `pi`."""
        assert pi.ndim == 3, "expected three-dimensional policy"
        assert np.allclose(pi.sum(axis=2), 1), "policy not normalized"
        assert np.all(pi >= 0), "policy has negative probabilities"
        self.pi = pi

    def _predict(self, observation: th.Tensor, deterministic: bool = False):
        raise NotImplementedError("Should never be called as predict overridden.")

    def forward(
        self,
        observation: th.Tensor,
        deterministic: bool = False,
    ) -> NoReturn:
        raise NotImplementedError("Should never be called.")  # pragma: no cover

    def predict(
        self,
        observation: Union[np.ndarray, Mapping[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
        """Predict action to take in given state.

        Arguments follow SB3 naming convention as this is an SB3 policy.
        In this convention, observations are returned by the environment,
        and state is a hidden state used by the policy (used by us to
        keep track of timesteps).

        What is `observation` here is a state in the underlying MDP,
        and would be called `state` elsewhere in this file.

        Args:
            observation: States in the underlying MDP.
            state: Hidden states of the policy -- used to represent timesteps by us.
            episode_start: Has episode completed?
            deterministic: If true, pick action with highest probability; otherwise,
                sample.

        Returns:
            Tuple of the actions and new hidden states.
        """
        if state is None:
            timesteps = np.zeros(len(observation), dtype=int)
        else:
            assert len(state) == 1
            timesteps = state[0]
        assert len(timesteps) == len(observation), "timestep and obs batch size differ"

        if episode_start is not None:
            timesteps[episode_start] = 0

        actions: List[int] = []
        for obs, t in zip(observation, timesteps):
            assert self.observation_space.contains(obs), "illegal state"
            dist = self.pi[t, obs, :]
            if deterministic:
                actions.append(int(dist.argmax()))
            else:
                actions.append(self.rng.choice(len(dist), p=dist))

        timesteps += 1  # increment timestep
        state = (timesteps,)
        return np.array(actions), state


MCEDemonstrations = Union[np.ndarray, base.AnyTransitions]


class MCEIRL(base.DemonstrationAlgorithm[types.TransitionsMinimal]):
    """Tabular MCE IRL.

    Reward is a function of observations, but policy is a function of states.

    The "observations" effectively exist just to let MCE IRL learn a reward
    in a reasonable feature space, giving a helpful inductive bias, e.g. that
    similar states have similar reward.

    Since we are performing planning to compute the policy, there is no need
    for function approximation in the policy.
    """

    demo_state_om: Optional[np.ndarray]

    def __init__(
        self,
        demonstrations: Optional[MCEDemonstrations],
        env: base_envs.TabularModelPOMDP,
        reward_net: reward_nets.RewardNet,
        rng: np.random.Generator,
        optimizer_cls: Type[th.optim.Optimizer] = th.optim.Adam,
        optimizer_kwargs: Optional[Mapping[str, Any]] = None,
        discount: float = 1.0,
        linf_eps: float = 1e-3,
        grad_l2_eps: float = 1e-4,
        # TODO(adam): do we need log_interval or can just use record_mean...?
        log_interval: Optional[int] = 100,
        *,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
    ) -> None:
        r"""Creates MCE IRL.

        Args:
            demonstrations: Demonstrations from an expert (optional). Can be a sequence
                of trajectories, or transitions, an iterable over mappings that
                represent a batch of transitions, or a state occupancy measure.
                The demonstrations must have observations one-hot coded unless
                demonstrations is a state-occupancy measure.
            env: a tabular MDP.
            rng: random state used for sampling from policy.
            reward_net: a neural network that computes rewards for the supplied
                observations.
            optimizer_cls: optimizer to use for supervised training.
            optimizer_kwargs: keyword arguments for optimizer construction.
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

        Raises:
            ValueError: if the env horizon is not finite (or an integer).
        """
        self.discount = discount
        self.env = env
        self.demo_state_om = None
        super().__init__(
            demonstrations=demonstrations,
            custom_logger=custom_logger,
        )

        self.reward_net = reward_net
        optimizer_kwargs = optimizer_kwargs or {"lr": 1e-2}
        self.optimizer = optimizer_cls(reward_net.parameters(), **optimizer_kwargs)

        self.linf_eps = linf_eps
        self.grad_l2_eps = grad_l2_eps
        self.log_interval = log_interval
        self.rng = rng

        # Initialize policy to be uniform random. We don't use this for MCE IRL
        # training, but it gives us something to return at all times with `policy`
        # property, similar to other algorithms.
        if self.env.horizon is None:
            raise ValueError("Only finite-horizon environments are supported.")
        ones = np.ones((self.env.horizon, self.env.state_dim, self.env.action_dim))
        uniform_pi = ones / self.env.action_dim
        self._policy = TabularPolicy(
            state_space=self.env.state_space,
            action_space=self.env.action_space,
            pi=uniform_pi,
            rng=self.rng,
        )

    def _set_demo_from_trajectories(self, trajs: Iterable[types.Trajectory]) -> None:
        self.demo_state_om = np.zeros((self.env.state_dim,))
        num_demos = 0
        for traj in trajs:
            cum_discount = 1.0
            for obs in types.assert_not_dictobs(traj.obs):
                self.demo_state_om[obs] += cum_discount
                cum_discount *= self.discount
            num_demos += 1
        self.demo_state_om /= num_demos

    def _set_demo_from_obs(
        self,
        obses: np.ndarray,
        dones: Optional[np.ndarray],
        next_obses: Optional[np.ndarray],
    ) -> None:
        self.demo_state_om = np.zeros((self.env.state_dim,))

        for obs in obses:
            if isinstance(obs, th.Tensor):
                obs = obs.item()  # must be scalar
            self.demo_state_om[obs] += 1.0

        # We assume the transitions were flattened from some trajectories,
        # then possibly shuffled. So add next observations for terminal states,
        # as they will not appear anywhere else; but ignore next observations
        # for all other states as they occur elsewhere in dataset.
        if dones is not None and next_obses is not None:
            for done, obs in zip(dones, next_obses):
                if isinstance(done, th.Tensor):
                    done = done.item()  # must be scalar
                    obs = obs.item()  # must be scalar
                if done:
                    self.demo_state_om[obs] += 1.0
        else:
            warnings.warn(
                "Training MCEIRL with transitions that lack next observation."
                "This will result in systematically wrong occupancy measure estimates.",
            )

        # Normalize occupancy measure estimates
        assert self.env.horizon is not None
        self.demo_state_om *= (self.env.horizon + 1) / self.demo_state_om.sum()

    def set_demonstrations(self, demonstrations: MCEDemonstrations) -> None:
        if isinstance(demonstrations, np.ndarray):
            # Demonstrations are an occupancy measure
            assert demonstrations.ndim == 1
            self.demo_state_om = demonstrations
            return

        # Demonstrations are either trajectories or transitions;
        # we must compute occupancy measure from this.
        if isinstance(demonstrations, Iterable):
            first_item, demonstrations_it = util.get_first_iter_element(demonstrations)
            if isinstance(first_item, types.Trajectory):
                self._set_demo_from_trajectories(demonstrations_it)
                return

        # Demonstrations are from some kind of transitions-like object. This does
        # not contain timesteps, so can only compute OM when undiscounted.
        if self.discount != 1.0:
            raise ValueError(
                "Cannot compute discounted OM from timeless Transitions.",
            )

        if isinstance(demonstrations, types.Transitions):
            self._set_demo_from_obs(
                types.assert_not_dictobs(demonstrations.obs),
                demonstrations.dones,
                types.assert_not_dictobs(demonstrations.next_obs),
            )
        elif isinstance(demonstrations, types.TransitionsMinimal):
            self._set_demo_from_obs(
                types.assert_not_dictobs(demonstrations.obs),
                None,
                None,
            )
        elif isinstance(demonstrations, Iterable):
            # Demonstrations are a Torch DataLoader or other Mapping iterable
            # Collect them together into one big NumPy array. This is inefficient,
            # we could compute the running statistics instead, but in practice do
            # not expect large dataset sizes together with MCE IRL.
            collated_list = collections.defaultdict(list)
            for batch in demonstrations:
                assert isinstance(batch, Mapping)
                for k in ("obs", "dones", "next_obs"):
                    if k in batch:
                        collated_list[k].append(batch[k])
            collated = {k: np.concatenate(v) for k, v in collated_list.items()}

            assert "obs" in collated
            for k, v in collated.items():
                assert len(v) == len(collated["obs"]), k
            self._set_demo_from_obs(
                collated["obs"],
                collated.get("dones"),
                collated.get("next_obs"),
            )
        else:
            raise TypeError(
                f"Unsupported demonstration type {type(demonstrations)}",
            )

    def _train_step(self, obs_mat: th.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        self.optimizer.zero_grad()

        # get reward predicted for each state by current model, & compute
        # expected # of times each state is visited by soft-optimal policy
        # w.r.t that reward function
        # TODO(adam): support not just state-only reward?
        predicted_r = squeeze_r(self.reward_net(obs_mat, None, None, None))
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
            dtype=self.reward_net.dtype,
            device=self.reward_net.device,
        )
        # The "loss" is then:
        #   E_\pi[r_\theta(S)] - E_D[r_\theta(S)]
        loss = th.dot(weights_th, predicted_r)
        # This gives the required gradient:
        #   E_\pi[\nabla r_\theta(S)] - E_D[\nabla r_\theta(S)].
        loss.backward()
        self.optimizer.step()

        return predicted_r_np, visitations

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
        torch_obs_mat = th.as_tensor(
            obs_mat,
            dtype=self.reward_net.dtype,
            device=self.reward_net.device,
        )
        assert self.demo_state_om is not None
        assert self.demo_state_om.shape == (len(obs_mat),)

        with networks.training(self.reward_net):
            # switch to training mode (affects dropout, normalization)
            for t in range(max_iter):
                predicted_r_np, visitations = self._train_step(torch_obs_mat)

                # these are just for termination conditions & debug logging
                grads = []
                for p in self.reward_net.parameters():
                    assert p.grad is not None  # for type checker
                    grads.append(p.grad)
                grad_norm = util.tensor_iter_norm(grads).item()
                linf_delta = np.max(np.abs(self.demo_state_om - visitations))

                if self.log_interval is not None and 0 == (t % self.log_interval):
                    params = self.reward_net.parameters()
                    weight_norm = util.tensor_iter_norm(params).item()
                    self.logger.record("iteration", t)
                    self.logger.record("linf_delta", linf_delta)
                    self.logger.record("weight_norm", weight_norm)
                    self.logger.record("grad_norm", grad_norm)
                    self.logger.dump(t)

                if linf_delta <= self.linf_eps or grad_norm <= self.grad_l2_eps:
                    break

        _, _, pi = mce_partition_fh(
            self.env,
            reward=predicted_r_np,
            discount=self.discount,
        )
        # TODO(adam): this policy works on states, not observations, so can't
        # actually compute rollouts from it in the usual way. Fix this by making
        # observations part of MCE IRL and turn environment from POMDP->MDP?
        self._policy.set_pi(pi)

        return visitations

    @property
    def policy(self) -> policies.BasePolicy:
        return self._policy
