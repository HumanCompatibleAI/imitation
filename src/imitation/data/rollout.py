"""Methods to collect, analyze and manipulate transition and trajectory rollouts."""

import collections
import dataclasses
import logging
from typing import (
    Any,
    Callable,
    Dict,
    Hashable,
    Iterable,
    List,
    Mapping,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import numpy as np
from gym import spaces
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.policies import BasePolicy
from stable_baselines3.common.utils import check_for_correct_spaces
from stable_baselines3.common.vec_env import VecEnv

from imitation.data import types


def unwrap_traj(traj: types.TrajectoryWithRew) -> types.TrajectoryWithRew:
    """Uses `RolloutInfoWrapper`-captured `obs` and `rews` to replace fields.

    This can be useful for bypassing other wrappers to retrieve the original
    `obs` and `rews`.

    Fails if `infos` is None or if the trajectory was generated from an
    environment without imitation.data.wrappers.RolloutInfoWrapper

    Args:
        traj: A trajectory generated from `RolloutInfoWrapper`-wrapped Environments.

    Returns:
        A copy of `traj` with replaced `obs` and `rews` fields.

    Raises:
        ValueError: If `traj.infos` is None
    """
    if traj.infos is None:
        raise ValueError("Trajectory must have infos to unwrap")
    ep_info = traj.infos[-1]["rollout"]
    res = dataclasses.replace(traj, obs=ep_info["obs"], rews=ep_info["rews"])
    assert len(res.obs) == len(res.acts) + 1
    assert len(res.rews) == len(res.acts)
    return res


class TrajectoryAccumulator:
    """Accumulates trajectories step-by-step.

    Useful for collecting completed trajectories while ignoring partially-completed
    trajectories (e.g. when rolling out a VecEnv to collect a set number of
    transitions). Each in-progress trajectory is identified by a 'key', which enables
    several independent trajectories to be collected at once. They key can also be left
    at its default value of `None` if you only wish to collect one trajectory.
    """

    def __init__(self):
        """Initialise the trajectory accumulator."""
        self.partial_trajectories = collections.defaultdict(list)

    def add_step(
        self,
        step_dict: Mapping[str, Union[np.ndarray, Mapping[str, Any], types.DictObs]],
        key: Hashable = None,
    ) -> None:
        """Add a single step to the partial trajectory identified by `key`.

        Generally a single step could correspond to, e.g., one environment managed
        by a VecEnv.

        Args:
            step_dict: dictionary containing information for the current step. Its
                keys could include any (or all) attributes of a `TrajectoryWithRew`
                (e.g. "obs", "acts", etc.).
            key: key to uniquely identify the trajectory to append to, if working
                with multiple partial trajectories.
        """
        self.partial_trajectories[key].append(step_dict)

    def finish_trajectory(
        self,
        key: Hashable,
        terminal: bool,
    ) -> types.TrajectoryWithRew:
        """Complete the trajectory labelled with `key`.

        Args:
            key: key uniquely identifying which in-progress trajectory to remove.
            terminal: trajectory has naturally finished (i.e. includes terminal state).

        Returns:
            traj: list of completed trajectories popped from
                `self.partial_trajectories`.
        """
        part_dicts = self.partial_trajectories[key]
        del self.partial_trajectories[key]
        out_dict_unstacked = collections.defaultdict(list)
        for part_dict in part_dicts:
            for k, array in part_dict.items():
                out_dict_unstacked[k].append(array)

        traj = types.TrajectoryWithRew(
            obs=types.stack_maybe_dictobs(out_dict_unstacked["obs"]),
            acts=np.stack(out_dict_unstacked["acts"], axis=0),
            infos=np.stack(out_dict_unstacked["infos"], axis=0),  # array of dict objs
            rews=np.stack(out_dict_unstacked["rews"], axis=0),
            terminal=terminal,
        )
        assert traj.rews.shape[0] == traj.acts.shape[0] == len(traj.obs) - 1
        return traj

    def add_steps_and_auto_finish(
        self,
        acts: np.ndarray,
        obs: Union[np.ndarray, Dict[str, np.ndarray], types.DictObs],
        rews: np.ndarray,
        dones: np.ndarray,
        infos: List[dict],
    ) -> List[types.TrajectoryWithRew]:
        """Calls `add_step` repeatedly using acts and the returns from `venv.step`.

        Also automatically calls `finish_trajectory()` for each `done == True`.
        Before calling this method, each environment index key needs to be
        initialized with the initial observation (usually from `venv.reset()`).

        See the body of `util.rollout.generate_trajectory` for an example.

        Args:
            acts: Actions passed into `VecEnv.step()`.
            obs: Return value from `VecEnv.step(acts)`.
            rews: Return value from `VecEnv.step(acts)`.
            dones: Return value from `VecEnv.step(acts)`.
            infos: Return value from `VecEnv.step(acts)`.

        Returns:
            A list of completed trajectories. There should be one trajectory for
            each `True` in the `dones` argument.
        """
        trajs: List[types.TrajectoryWithRew] = []
        wrapped_obs = types.DictObs.maybe_wrap(obs)

        # len of dictobs is the shape[0] of each value array - which here is # of envs
        for env_idx in range(len(wrapped_obs)):
            assert env_idx in self.partial_trajectories
            assert list(self.partial_trajectories[env_idx][0].keys()) == ["obs"], (
                "Need to first initialize partial trajectory using "
                "self._traj_accum.add_step({'obs': ob}, key=env_idx)"
            )

        zip_iter = enumerate(zip(acts, wrapped_obs, rews, dones, infos))
        for env_idx, (act, ob, rew, done, info) in zip_iter:
            if done:
                # When dones[i] from VecEnv.step() is True, obs[i] is the first
                # observation following reset() of the ith VecEnv, and
                # infos[i]["terminal_observation"] is the actual final observation.
                real_ob = info["terminal_observation"]
                if isinstance(real_ob, dict):
                    # TODO: does this need to be unsqueezed or something?
                    real_ob = types.DictObs(real_ob)
            else:
                real_ob = ob

            self.add_step(
                dict(
                    acts=act,
                    rews=rew,
                    # this is not the obs corresponding to `act`, but rather the obs
                    # *after* `act` (see above)
                    obs=real_ob,
                    infos=info,
                ),
                env_idx,
            )
            if done:
                # finish env_idx-th trajectory
                new_traj = self.finish_trajectory(env_idx, terminal=True)
                trajs.append(new_traj)
                # When done[i] from VecEnv.step() is True, obs[i] is the first
                # observation following reset() of the ith VecEnv.
                self.add_step(dict(obs=ob), env_idx)
        return trajs


GenTrajTerminationFn = Callable[[Sequence[types.TrajectoryWithRew]], bool]


def make_min_episodes(n: int) -> GenTrajTerminationFn:
    """Terminate after collecting n episodes of data.

    Args:
        n: Minimum number of episodes of data to collect.
            May overshoot if two episodes complete simultaneously (unlikely).

    Returns:
        A function implementing this termination condition.
    """
    assert n >= 1
    return lambda trajectories: len(trajectories) >= n


def make_min_timesteps(n: int) -> GenTrajTerminationFn:
    """Terminate at the first episode after collecting n timesteps of data.

    Args:
        n: Minimum number of timesteps of data to collect.
            May overshoot to nearest episode boundary.

    Returns:
        A function implementing this termination condition.
    """
    assert n >= 1

    def f(trajectories: Sequence[types.TrajectoryWithRew]):
        timesteps = sum(len(t.obs) - 1 for t in trajectories)
        return timesteps >= n

    return f


def make_sample_until(
    min_timesteps: Optional[int] = None,
    min_episodes: Optional[int] = None,
) -> GenTrajTerminationFn:
    """Returns a termination condition sampling for a number of timesteps and episodes.

    Args:
        min_timesteps: Sampling will not stop until there are at least this many
            timesteps.
        min_episodes: Sampling will not stop until there are at least this many
            episodes.

    Returns:
        A termination condition.

    Raises:
        ValueError: Neither of n_timesteps and n_episodes are set, or either are
            non-positive.
    """
    if min_timesteps is None and min_episodes is None:
        raise ValueError(
            "At least one of min_timesteps and min_episodes needs to be non-None",
        )

    conditions = []
    if min_timesteps is not None:
        if min_timesteps <= 0:
            raise ValueError(
                f"min_timesteps={min_timesteps} if provided must be positive",
            )
        conditions.append(make_min_timesteps(min_timesteps))

    if min_episodes is not None:
        if min_episodes <= 0:
            raise ValueError(
                f"min_episodes={min_episodes} if provided must be positive",
            )
        conditions.append(make_min_episodes(min_episodes))

    def sample_until(trajs: Sequence[types.TrajectoryWithRew]) -> bool:
        for cond in conditions:
            if not cond(trajs):
                return False
        return True

    return sample_until


# A PolicyCallable is a function that takes an array of observations, an optional
# array of states, and an optional array of episode starts and returns an array of
# corresponding actions.
PolicyCallable = Callable[
    [
        Union[np.ndarray, types.DictObs],
        Optional[Tuple[np.ndarray, ...]],
        Optional[np.ndarray],
    ],
    Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]],
]
AnyPolicy = Union[BaseAlgorithm, BasePolicy, PolicyCallable, None]


def policy_to_callable(
    policy: AnyPolicy,
    venv: VecEnv,
    deterministic_policy: bool = False,
) -> PolicyCallable:
    """Converts any policy-like object into a function from observations to actions."""
    get_actions: PolicyCallable
    if policy is None:

        def get_actions(
            observations: Union[np.ndarray, types.DictObs],
            states: Optional[Tuple[np.ndarray, ...]],
            episode_starts: Optional[np.ndarray],
        ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
            acts = [venv.action_space.sample() for _ in range(len(observations))]
            return np.stack(acts, axis=0), None

    elif isinstance(policy, (BaseAlgorithm, BasePolicy)):
        # There's an important subtlety here: BaseAlgorithm and BasePolicy
        # are themselves Callable (which we check next). But in their case,
        # we want to use the .predict() method, rather than __call__()
        # (which would call .forward()). So this elif clause must come first!

        def get_actions(
            observations: Union[np.ndarray, types.DictObs],
            states: Optional[Tuple[np.ndarray, ...]],
            episode_starts: Optional[np.ndarray],
        ) -> Tuple[np.ndarray, Optional[Tuple[np.ndarray, ...]]]:
            assert isinstance(policy, (BaseAlgorithm, BasePolicy))
            # pytype doesn't seem to understand that policy is a BaseAlgorithm
            # or BasePolicy here, rather than a Callable
            (acts, states) = policy.predict(  # pytype: disable=attribute-error
                types.maybe_unwrap_dictobs(observations),
                state=states,
                episode_start=episode_starts,
                deterministic=deterministic_policy,
            )
            return acts, states

    elif callable(policy):
        # When a policy callable is passed, by default we will use it directly.
        # We are not able to change the determinism of the policy when it is a
        # callable that only takes in the states.
        if deterministic_policy:
            raise ValueError(
                "Cannot set deterministic_policy=True when policy is a callable, "
                "since deterministic_policy argument is ignored.",
            )
        get_actions = policy

    else:
        raise TypeError(
            "Policy must be None, a stable-baselines policy or algorithm, "
            f"or a Callable, got {type(policy)} instead",
        )

    if isinstance(policy, BaseAlgorithm):
        # Check that the observation and action spaces of policy and environment match
        try:
            check_for_correct_spaces(
                venv,
                policy.observation_space,
                policy.action_space,
            )
        except ValueError as e:
            # Check for a particularly common mistake when using image environments.
            venv_obs_shape = venv.observation_space.shape
            assert policy.observation_space is not None
            policy_obs_shape = policy.observation_space.shape
            if len(venv_obs_shape) != 3 or len(policy_obs_shape) != 3:
                raise e
            venv_obs_rearranged = (
                venv_obs_shape[2],
                venv_obs_shape[0],
                venv_obs_shape[1],
            )
            if venv_obs_rearranged != policy_obs_shape:
                raise e
            raise ValueError(
                "Policy and environment observation shape mismatch. "
                "This is likely caused by "
                "https://github.com/HumanCompatibleAI/imitation/issues/599. "
                "If encountering this from rollout.rollout, try calling:\n"
                "rollout.rollout(expert, expert.get_env(), ...) instead of\n"
                "rollout.rollout(expert, env, ...)\n\n"
                f"Policy observation shape: {policy_obs_shape} \n"
                f"Environment observation shape: {venv_obs_shape}",
            )

    return get_actions


def generate_trajectories(
    policy: AnyPolicy,
    venv: VecEnv,
    sample_until: GenTrajTerminationFn,
    rng: np.random.Generator,
    *,
    deterministic_policy: bool = False,
) -> Sequence[types.TrajectoryWithRew]:
    """Generate trajectory dictionaries from a policy and an environment.

    Args:
        policy: Can be any of the following:
            1) A stable_baselines3 policy or algorithm trained on the gym environment.
            2) A Callable that takes an ndarray of observations and returns an ndarray
            of corresponding actions.
            3) None, in which case actions will be sampled randomly.
        venv: The vectorized environments to interact with.
        sample_until: A function determining the termination condition.
            It takes a sequence of trajectories, and returns a bool.
            Most users will want to use one of `min_episodes` or `min_timesteps`.
        deterministic_policy: If True, asks policy to deterministically return
            action. Note the trajectories might still be non-deterministic if the
            environment has non-determinism!
        rng: used for shuffling trajectories.

    Returns:
        Sequence of trajectories, satisfying `sample_until`. Additional trajectories
        may be collected to avoid biasing process towards short episodes; the user
        should truncate if required.
    """
    get_actions = policy_to_callable(policy, venv, deterministic_policy)

    # Collect rollout tuples.
    trajectories = []
    # accumulator for incomplete trajectories
    trajectories_accum = TrajectoryAccumulator()
    obs = venv.reset()

    assert isinstance(
        obs,
        (
            np.ndarray,
            dict,
        ),
    ), "Tuple observations are not supported."

    # need to wrap here to iterate over envs properly
    wrapped_obs = types.DictObs.maybe_wrap(obs)
    for env_idx, ob in enumerate(wrapped_obs):
        # Seed with first obs only. Inside loop, we'll only add second obs from
        # each (s,a,r,s') tuple, under the same "obs" key again. That way we still
        # get all observations, but they're not duplicated into "next obs" and
        # "previous obs" (this matters for, e.g., Atari, where observations are
        # really big).
        trajectories_accum.add_step(dict(obs=ob), env_idx)

    # Now, we sample until `sample_until(trajectories)` is true.
    # If we just stopped then this would introduce a bias towards shorter episodes,
    # since longer episodes are more likely to still be active, i.e. in the process
    # of being sampled from. To avoid this, we continue sampling until all epsiodes
    # are complete.
    #
    # To start with, all environments are active.
    active = np.ones(venv.num_envs, dtype=bool)
    state = None
    dones = np.zeros(venv.num_envs, dtype=bool)
    while np.any(active):
        acts, state = get_actions(wrapped_obs, state, dones)
        obs, rews, dones, infos = venv.step(acts)
        assert isinstance(
            obs,
            (np.ndarray, dict),
        ), "Tuple observations are not supported."
        wrapped_obs = types.DictObs.maybe_wrap(obs)

        # If an environment is inactive, i.e. the episode completed for that
        # environment after `sample_until(trajectories)` was true, then we do
        # *not* want to add any subsequent trajectories from it. We avoid this
        # by just making it never done.
        dones &= active

        new_trajs = trajectories_accum.add_steps_and_auto_finish(
            acts,
            wrapped_obs,
            rews,
            dones,
            infos,
        )
        trajectories.extend(new_trajs)

        if sample_until(trajectories):
            # Termination condition has been reached. Mark as inactive any
            # environments where a trajectory was completed this timestep.
            active &= ~dones

    # Note that we just drop partial trajectories. This is not ideal for some
    # algos; e.g. BC can probably benefit from partial trajectories, too.

    # Each trajectory is sampled i.i.d.; however, shorter episodes are added to
    # `trajectories` sooner. Shuffle to avoid bias in order. This is important
    # when callees end up truncating the number of trajectories or transitions.
    # It is also cheap, since we're just shuffling pointers.
    rng.shuffle(trajectories)  # type: ignore[arg-type]

    # Sanity checks.
    for trajectory in trajectories:
        n_steps = len(trajectory.acts)
        # extra 1 for the end
        if isinstance(venv.observation_space, spaces.dict.Dict):
            exp_obs = {
                k: (n_steps + 1,) + v.shape for k, v in venv.observation_space.items()
            }
        else:
            exp_obs = (n_steps + 1,) + venv.observation_space.shape
        real_obs = trajectory.obs.shape
        assert real_obs == exp_obs, f"expected shape {exp_obs}, got {real_obs}"
        exp_act = (n_steps,) + venv.action_space.shape
        real_act = trajectory.acts.shape
        assert real_act == exp_act, f"expected shape {exp_act}, got {real_act}"
        exp_rew = (n_steps,)
        real_rew = trajectory.rews.shape
        assert real_rew == exp_rew, f"expected shape {exp_rew}, got {real_rew}"

    return trajectories


def rollout_stats(
    trajectories: Sequence[types.TrajectoryWithRew],
) -> Mapping[str, float]:
    """Calculates various stats for a sequence of trajectories.

    Args:
        trajectories: Sequence of trajectories.

    Returns:
        Dictionary containing `n_traj` collected (int), along with episode return
        statistics (keys: `{monitor_,}return_{min,mean,std,max}`, float values)
        and trajectory length statistics (keys: `len_{min,mean,std,max}`, float
        values).

        `return_*` values are calculated from environment rewards.
        `monitor_*` values are calculated from Monitor-captured rewards, and
        are only included if the `trajectories` contain Monitor infos.
    """
    assert len(trajectories) > 0
    out_stats: Dict[str, float] = {"n_traj": len(trajectories)}
    traj_descriptors = {
        "return": np.asarray([sum(t.rews) for t in trajectories]),
        "len": np.asarray([len(t.rews) for t in trajectories]),
    }

    monitor_ep_returns = []
    for t in trajectories:
        if t.infos is not None:
            ep_return = t.infos[-1].get("episode", {}).get("r")
            if ep_return is not None:
                monitor_ep_returns.append(ep_return)
    if monitor_ep_returns:
        # Note monitor_ep_returns[i] may be from a different episode than ep_return[i]
        # since we skip episodes with None infos. This is OK as we only return summary
        # statistics, but you cannot e.g. compute the correlation between ep_return and
        # monitor_ep_returns.
        traj_descriptors["monitor_return"] = np.asarray(monitor_ep_returns)
        # monitor_return_len may be < n_traj when infos is sometimes missing
        out_stats["monitor_return_len"] = len(traj_descriptors["monitor_return"])

    stat_names = ["min", "mean", "std", "max"]
    for desc_name, desc_vals in traj_descriptors.items():
        for stat_name in stat_names:
            stat_value: np.generic = getattr(np, stat_name)(desc_vals)
            # Convert numpy type to float or int. The numpy operators always return
            # a numpy type, but we want to return type float. (int satisfies
            # float type for the purposes of static-typing).
            out_stats[f"{desc_name}_{stat_name}"] = stat_value.item()

    for v in out_stats.values():
        assert isinstance(v, (int, float))
    return out_stats


def flatten_trajectories(
    trajectories: Iterable[types.Trajectory],
) -> types.Transitions:
    """Flatten a series of trajectory dictionaries into arrays.

    Args:
        trajectories: list of trajectories.

    Returns:
        The trajectories flattened into a single batch of Transitions.
    """

    def all_of_type(key, desired_type):
        return all(
            isinstance(getattr(traj, key), desired_type) for traj in trajectories
        )

    assert all_of_type("obs", types.DictObs) or all_of_type("obs", np.ndarray)
    assert all_of_type("acts", np.ndarray)

    # sad to use Any here, but mypy struggles otherwise.
    # we enforce type constraints in asserts above and below.
    keys = ["obs", "next_obs", "acts", "dones", "infos"]
    parts: Mapping[str, List[Any]] = {key: [] for key in keys}
    for traj in trajectories:
        parts["acts"].append(traj.acts)

        obs = traj.obs

        parts["obs"].append(obs[:-1])
        parts["next_obs"].append(obs[1:])

        dones = np.zeros(len(traj.acts), dtype=bool)
        dones[-1] = traj.terminal
        parts["dones"].append(dones)

        if traj.infos is None:
            infos = np.array([{}] * len(traj))
        else:
            infos = traj.infos
        parts["infos"].append(infos)

    cat_parts = {
        key: types.concatenate_maybe_dictobs(part_list)
        for key, part_list in parts.items()
    }
    lengths = set(map(len, cat_parts.values()))
    assert len(lengths) == 1, f"expected one length, got {lengths}"
    return types.Transitions(**cat_parts)


def flatten_trajectories_with_rew(
    trajectories: Sequence[types.TrajectoryWithRew],
) -> types.TransitionsWithRew:
    transitions = flatten_trajectories(trajectories)
    rews = np.concatenate([traj.rews for traj in trajectories])
    return types.TransitionsWithRew(**dataclasses.asdict(transitions), rews=rews)


def generate_transitions(
    policy: AnyPolicy,
    venv: VecEnv,
    n_timesteps: int,
    rng: np.random.Generator,
    *,
    truncate: bool = True,
    **kwargs: Any,
) -> types.TransitionsWithRew:
    """Generate obs-action-next_obs-reward tuples.

    Args:
        policy: Can be any of the following:
            - A stable_baselines3 policy or algorithm trained on the gym environment
            - A Callable that takes an ndarray of observations and returns an ndarray
            of corresponding actions
            - None, in which case actions will be sampled randomly
        venv: The vectorized environments to interact with.
        n_timesteps: The minimum number of timesteps to sample.
        rng: The random state to use for sampling trajectories.
        truncate: If True, then drop any additional samples to ensure that exactly
            `n_timesteps` samples are returned.
        **kwargs: Passed-through to generate_trajectories.

    Returns:
        A batch of Transitions. The length of the constituent arrays is guaranteed
        to be at least `n_timesteps` (if specified), but may be greater unless
        `truncate` is provided as we collect data until the end of each episode.
    """
    traj = generate_trajectories(
        policy,
        venv,
        sample_until=make_min_timesteps(n_timesteps),
        rng=rng,
        **kwargs,
    )
    transitions = flatten_trajectories_with_rew(traj)
    if truncate and n_timesteps is not None:
        as_dict = dataclasses.asdict(transitions)
        truncated = {k: arr[:n_timesteps] for k, arr in as_dict.items()}
        transitions = types.TransitionsWithRew(**truncated)
    return transitions


def rollout(
    policy: AnyPolicy,
    venv: VecEnv,
    sample_until: GenTrajTerminationFn,
    rng: np.random.Generator,
    *,
    unwrap: bool = True,
    exclude_infos: bool = True,
    verbose: bool = True,
    **kwargs: Any,
) -> Sequence[types.TrajectoryWithRew]:
    """Generate policy rollouts.

    This method is a wrapper of generate_trajectories that allows
    the user to additionally replace the rewards and observations with the original
    values if the environment is wrapped, to exclude the infos from the
    trajectories, and to print summary statistics of the rollout.

    The `.infos` field of each Trajectory is set to `None` to save space.

    Args:
        policy: Can be any of the following:
            1) A stable_baselines3 policy or algorithm trained on the gym environment.
            2) A Callable that takes an ndarray of observations and returns an ndarray
            of corresponding actions.
            3) None, in which case actions will be sampled randomly.
        venv: The vectorized environments.
        sample_until: End condition for rollout sampling.
        rng: Random state to use for sampling.
        unwrap: If True, then save original observations and rewards (instead of
            potentially wrapped observations and rewards) by calling
            `unwrap_traj()`.
        exclude_infos: If True, then exclude `infos` from pickle by setting
            this field to None. Excluding `infos` can save a lot of space during
            pickles.
        verbose: If True, then print out rollout stats before saving.
        **kwargs: Passed through to `generate_trajectories`.

    Returns:
        Sequence of trajectories, satisfying `sample_until`. Additional trajectories
        may be collected to avoid biasing process towards short episodes; the user
        should truncate if required.
    """
    trajs = generate_trajectories(
        policy,
        venv,
        sample_until,
        rng=rng,
        **kwargs,
    )
    if unwrap:
        trajs = [unwrap_traj(traj) for traj in trajs]
    if exclude_infos:
        trajs = [dataclasses.replace(traj, infos=None) for traj in trajs]
    if verbose:
        stats = rollout_stats(trajs)
        logging.info(f"Rollout stats: {stats}")
    return trajs


def discounted_sum(arr: np.ndarray, gamma: float) -> Union[np.ndarray, float]:
    """Calculate the discounted sum of `arr`.

    If `arr` is an array of rewards, then this computes the return;
    however, it can also be used to e.g. compute discounted state
    occupancy measures.

    Args:
        arr: 1 or 2-dimensional array to compute discounted sum over.
            Last axis is timestep, from current time step (first) to
            last timestep (last). First axis (if present) is batch
            dimension.
        gamma: the discount factor used.

    Returns:
        The discounted sum over the timestep axis. The first timestep is undiscounted,
        i.e. we start at gamma^0.
    """
    # We want to calculate sum_{t = 0}^T gamma^t r_t, which can be
    # interpreted as the polynomial sum_{t = 0}^T r_t x^t
    # evaluated at x=gamma.
    # Compared to first computing all the powers of gamma, then
    # multiplying with the `arr` values and then summing, this method
    # should require fewer computations and potentially be more
    # numerically stable.
    assert arr.ndim in (1, 2)
    if gamma == 1.0:
        return arr.sum(axis=0)
    else:
        return np.polynomial.polynomial.polyval(gamma, arr)
