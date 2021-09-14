"""Density-based baselines for imitation learning.

Each of these algorithms learns a density estimate on some aspect of the demonstrations,
then rewards the agent for following that estimate.
"""

import enum
import itertools
from typing import Iterable, Mapping, Optional, Sequence

import gym
import numpy as np
from gym.spaces.utils import flatten
from sklearn import neighbors, preprocessing
from stable_baselines3.common import on_policy_algorithm, vec_env

from imitation.algorithms import base
from imitation.data import rollout, types
from imitation.util import logger as imit_logger
from imitation.util import reward_wrapper

# Constants identifying different kinds of density we can use. Note that all
# can be augmented to depend on the time step by passing `is_stationary = True`
# to `DensityReward`.


class DensityType(enum.Enum):
    # Density on state s
    STATE_DENSITY = enum.auto()
    # Density on (s,a) pairs
    STATE_ACTION_DENSITY = enum.auto()
    # Density (s,s') pairs
    STATE_STATE_DENSITY = enum.auto()


class DensityReward(base.DemonstrationAlgorithm):
    transitions: Mapping[Optional[int], np.ndarray]

    def __init__(
        self,
        *,
        demonstrations: Optional[base.AnyTransitions],
        density_type: DensityType,
        kernel: str,
        kernel_bandwidth: float,
        obs_space: gym.Space,
        act_space: gym.Space,
        is_stationary: bool = True,
        standardise_inputs: bool = True,
        custom_logger: Optional[imit_logger.HierarchicalLogger] = None,
        allow_variable_horizon: bool = False,
    ):
        """Reward function based on a density estimate of trajectories.

        Args:
            demonstrations: expert demonstration trajectories.
            obs_space: observation space for underlying environment.
            act_space: action space for underlying environment.
            density_type: type of density to train on: single state, state-action pairs,
                or state-state pairs.
            standardise_inputs: if True, then the inputs to the reward model
                will be standardised to have zero mean and unit variance over the
                demonstration trajectories. Otherwise, inputs will be passed to the
                reward model with their ordinary scale.
            kernel: kernel to use for density estimation with `sklearn.KernelDensity`.
            kernel_bandwidth: bandwidth of kernel. If `standardise_inputs` is
                true and you are using a Gaussian kernel, then it probably makes sense
                to set this somewhere between 0.1 and 1.
            is_stationary: if True, share same density models for all timesteps;
                if False, use a different density model for each timestep.
                A non-stationary model is particularly likely to be useful when using
                STATE_DENSITY, to encourage agent to imitate entire trajectories, not
                just a few states that have high frequency in the demonstration dataset.
                If non-stationary, demonstrations must be trajectories, not transitions
                (which do not contain timesteps).
            custom_logger: Where to log to; if None (default), creates a new logger.
            allow_variable_horizon: If False (default), algorithm will raise an
                exception if it detects trajectories of different length during
                training. If True, overrides this safety check. WARNING: variable
                horizon episodes leak information about the reward via termination
                condition, and can seriously confound evaluation. Read
                https://imitation.readthedocs.io/en/latest/guide/variable_horizon.html
                before overriding this.
        """
        self.is_stationary = is_stationary
        self.density_type = density_type
        self.obs_space = obs_space
        self.act_space = act_space
        self.transitions = {}
        super().__init__(
            demonstrations=demonstrations,
            custom_logger=custom_logger,
            allow_variable_horizon=allow_variable_horizon,
        )
        self.kernel = kernel
        self.kernel_bandwidth = kernel_bandwidth
        self.standardise = standardise_inputs
        self._scaler = None
        self._density_models = {}

    def set_demonstrations(self, demonstrations: base.AnyTransitions) -> None:
        self.transitions = {}

        if isinstance(demonstrations, Iterable):
            first_item = iter(demonstrations).__next__()
            if isinstance(first_item, types.Trajectory):
                # Demonstrations are trajectories.
                # We have timestep information.
                for traj in demonstrations:
                    for i, (obs, act, next_obs) in enumerate(
                        zip(traj.obs[:-1], traj.acts, traj.obs[1:])
                    ):
                        flat_trans = self._preprocess_transition(obs, act, next_obs)
                        self.transitions.setdefault(i, []).append(flat_trans)
            else:
                # Demonstrations are a Torch DataLoader or other Mapping iterable
                for batch in demonstrations:
                    next_obses = batch.get("next_obs", itertools.repeat(None))
                    for obs, act, next_obs in zip(
                        batch["obs"], batch["acts"], next_obses
                    ):
                        flat_trans = self._preprocess_transition(obs, act, next_obs)
                        self.transitions.setdefault(None, []).append(flat_trans)
        elif isinstance(demonstrations, types.TransitionsMinimal):
            next_obses = (
                demonstrations.next_obs
                if hasattr(demonstrations, "next_obs")
                else itertools.repeat(None)
            )
            for obs, act, next_obs in zip(
                demonstrations.obs, demonstrations.acts, next_obses
            ):
                flat_trans = self._preprocess_transition(obs, act, next_obs)
                self.transitions.setdefault(None, []).append(flat_trans)
        else:
            raise TypeError(f"Unsupported demonstration type {type(demonstrations)}")

        self.transitions = {k: np.stack(v, axis=0) for k, v in self.transitions.items()}

        if not self.is_stationary and None in self.transitions:
            raise ValueError(
                "Non-stationary model incompatible with non-trajectory demonstrations."
            )
        if self.is_stationary:
            self.transitions = {
                None: np.concatenate(list(self.transitions.values()), axis=0)
            }

    def train(self):
        # if requested, we'll scale demonstration transitions so that they have
        # zero mean and unit variance (i.e all components are equally important)
        self._scaler = preprocessing.StandardScaler(
            with_mean=self.standardise, with_std=self.standardise
        )
        flattened_dataset = np.concatenate(list(self.transitions.values()), axis=0)
        self._scaler.fit(flattened_dataset)

        # now fit density model
        self._density_models = {
            k: self._fit_density(self._scaler.transform(v))
            for k, v in self.transitions.items()
        }

    def _fit_density(self, transitions: np.ndarray) -> neighbors.KernelDensity:
        # This bandwidth was chosen to make sense with standardised inputs that
        # have unit variance in each component. There might be a better way to
        # choose it automatically.
        density_model = neighbors.KernelDensity(
            kernel=self.kernel, bandwidth=self.kernel_bandwidth
        )
        density_model.fit(transitions)
        return density_model

    def _preprocess_transition(
        self, obs: np.ndarray, act: np.ndarray, next_obs: np.ndarray
    ) -> np.ndarray:
        if self.density_type == DensityType.STATE_DENSITY:
            return flatten(self.obs_space, obs)
        elif self.density_type == DensityType.STATE_ACTION_DENSITY:
            return np.concatenate(
                [flatten(self.obs_space, obs), flatten(self.act_space, act)]
            )
        elif self.density_type == DensityType.STATE_STATE_DENSITY:
            return np.concatenate(
                [flatten(self.obs_space, obs), flatten(self.obs_space, next_obs)]
            )
        else:
            raise ValueError(f"Unknown density type {self.density_type}")

    def __call__(
        self,
        obs_b: np.ndarray,
        act_b: np.ndarray,
        next_obs_b: np.ndarray,
        dones: np.ndarray,
        steps: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        r"""Compute reward from given (s,a,s') transition batch.

        This handles *batches* of observations, since it's designed to work with
        VecEnvs.

        Args:
            obs_b: current batch of observations.
            act_b: batch of actions that agent took in response to those
                observations.
            next_obs_b: batch of observations encountered after the
                agent took those actions.
            dones: is it terminal state?
            steps: Optional -- what timestep is this from?

        Returns:
            Array of scalar rewards of the form `r_t(s,a,s') = \log \hat p_t(s,a,s')`
            (one for each environment), where `\log \hat p` is the underlying density
            model (and may be independent of s', a, or t, depending on options passed
            to constructor).
        """
        if not self.is_stationary and steps is None:
            raise ValueError("steps must be provided with non-stationary models")

        del dones  # TODO(adam): should we handle terminal state specially in any way?

        rew_list = []
        assert len(obs_b) == len(act_b) and len(obs_b) == len(next_obs_b)
        for idx, (obs, act, next_obs) in enumerate(zip(obs_b, act_b, next_obs_b)):
            flat_trans = self._preprocess_transition(obs, act, next_obs)
            scaled_padded_trans = self._scaler.transform(flat_trans[np.newaxis])
            if self.is_stationary:
                rew = self._density_models[None].score(scaled_padded_trans)
            else:
                time = steps[idx]
                if time >= len(self._density_models):
                    # Can't do anything sensible here yet. Correct solution is to use
                    # hierarchical model in which we first check whether state is
                    # absorbing, then assign either constant score or a score based on
                    # density.
                    raise ValueError(
                        f"Time {time} out of range (0, {len(self._density_models)}], "
                        "and absorbing states not currently supported"
                    )
                else:
                    time_model = self._density_models[time]
                    rew = time_model.score(scaled_padded_trans)
            rew_list.append(rew)
        rew_array = np.asarray(rew_list, dtype="float32")
        return rew_array


# TODO(adam): Do we even need this? Merge into one class perhaps?
class DensityTrainer:
    def __init__(
        self,
        venv: vec_env.VecEnv,
        rollouts: Sequence[types.Trajectory],
        imitation_trainer: on_policy_algorithm.OnPolicyAlgorithm,
        *,
        standardise_inputs: bool = True,
        kernel: str = "gaussian",
        kernel_bandwidth: float = 0.5,
        density_type: DensityType = DensityType.STATE_ACTION_DENSITY,
    ):
        r"""Family of simple imitation learning baseline algorithms that apply RL to
        maximise a rough density estimate of the demonstration trajectories.
        Specifically, it constructs a non-parametric estimate of `p(s)`, `p(s,s')`,
        `p_t(s,a)`, etc. (depending on options), then rewards the imitation learner
        with `r_t(s,a,s')=\log p_t(s,a,s')` (or `\log p(s,s')`, or whatever the
        user wants the model to condition on).

        Args:
            venv: environment to train on.
            rollouts: list of expert trajectories to imitate.
            imitation_trainer: RL algorithm & initial policy that will
                be used to train the imitation learner.
            kernel, kernel_bandwidth, density_type, is_stationary,
                standardise_inputs: these are passed directly to `DensityReward`;
                refer to documentation for that class."""
        self.venv = venv
        self.imitation_trainer = imitation_trainer
        self.reward_fn = DensityReward(
            demonstrations=rollouts,
            density_type=density_type,
            obs_space=self.venv.observation_space,
            act_space=self.venv.action_space,
            kernel=kernel,
            kernel_bandwidth=kernel_bandwidth,
            standardise_inputs=standardise_inputs,
        )
        self.wrapped_env = reward_wrapper.RewardVecEnvWrapper(self.venv, self.reward_fn)

    def train_policy(self, n_timesteps=int(1e6), **kwargs):
        """Train the imitation policy for a given number of timesteps.

        Args:
            n_timesteps (int): number of timesteps to train the policy for.
            kwargs (dict): extra arguments that will be passed to the `learn()`
                method of the imitation RL model. Refer to Stable Baselines docs for
                details.
        """
        self.reward_fn.train()
        self.imitation_trainer.set_env(self.wrapped_env)
        # FIXME: learn() is not meant to be called frequently; there are
        # significant per-call overheads (see Adam's comment in adversarial.py)
        # FIXME: the ep_reward_mean reported by SB is wrong; it comes from a
        # Monitor() that is being (incorrectly) used to wrap the underlying
        # environment.
        self.imitation_trainer.learn(
            n_timesteps,
            # ensure we can see total steps for all
            # learn() calls, not just for this call
            reset_num_timesteps=False,
            **kwargs,
        )

    def test_policy(self, *, n_trajectories=10, true_reward=True):
        """Test current imitation policy on environment & give some rollout stats.

        Args:
            n_trajectories (int): number of rolled-out trajectories.
            true_reward (bool): should this use ground truth reward from underlying
                environment (True), or imitation reward (False)?

        Returns:
            dict: rollout statistics collected by
                `imitation.utils.rollout.rollout_stats()`.
        """
        self.imitation_trainer.set_env(self.venv)
        trajs = rollout.generate_trajectories(
            self.imitation_trainer,
            self.venv if true_reward else self.wrapped_env,
            sample_until=rollout.make_min_episodes(n_trajectories),
        )
        reward_stats = rollout.rollout_stats(trajs)
        return reward_stats
