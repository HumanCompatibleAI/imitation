

import numpy as np
import tensorflow as tf

from . import util  # Relative import needed to prevent cycle with __init__.py


def generate(policy, env, *, n_timesteps=None, n_episodes=None,
             truncate_timesteps=False):
    """
    Generate old_obs-action-new_obs-reward tuples from a policy and an
    environment.

    Params:
    policy (stable_baselines.BaseRLModel) -- A stable_baselines Model, trained
      on the gym environment.
    env (VecEnv or Env or str) -- The environment(s) to interact with.
    n_timesteps (int) -- The number of obs-action-obs-reward tuples to collect.
      The `truncate_timesteps` parameter chooses whether to discard extra
      tuples.
      Set exactly one of `n_timesteps` and `n_episodes`, or this function will
      error.
    n_episodes (int) -- The number of episodes to finish before returning
      collected tuples. Tuples from parallel episodes underway when the final
      episode is finished will also be returned.
      Set exactly one of `n_timesteps` and `n_episodes`, or this function will
      error.
    truncate_timesteps (bool) -- If True, then discard any tuples, ensuring that
      exactly `n_timesteps` are returned. Otherwise, return every collected
      tuple.

    Return:
    rollout_obs_old (array) -- A numpy array with shape
      `[n_timesteps] + env.observation_space.shape`. The ith observation in this
      array is the observation seen with the agent chooses action
      `rollout_act[i]`.
    rollout_act (array) -- A numpy array with shape
      `[n_timesteps] + env.action_space.shape`.
    rollout_obs_new (array) -- A numpy array with shape
      `[n_timesteps] + env.observation_space.shape`. The ith observation in this
      array is from the transition state after the agent chooses action
      `rollout_act[i]`.
    rollout_rewards (array) -- A numpy array with shape `[n_timesteps]`. The
      reward received on the ith timestep is `rollout_rewards[i]`.
    """
    env = util.maybe_load_env(env, vectorize=True)
    policy.set_env(env)  # This checks that env and policy are compatbile.
    assert util.is_vec_env(env)

    # Validate end condition arguments and initialize end conditions.
    if n_timesteps is not None and n_episodes is not None:
        raise ValueError("n_timesteps and n_episodes were both set")
    elif n_timesteps is not None:
        assert n_timesteps > 0
        end_cond = "timesteps"
    elif n_episodes is not None:
        assert n_episodes > 0
        end_cond = "episodes"
        episodes_elapsed = 0
    else:
        raise ValueError("Set at least one of n_timesteps and n_episodes")

    # Implements end-condition logic.
    def rollout_done():
        if end_cond == "timesteps":
            return len(rollout_obs_new) >= n_timesteps
        elif end_cond == "episodes":
            return episodes_elapsed >= n_episodes
        else:
            raise RuntimeError(end_cond)

    # Collect rollout tuples.
    rollout_obs_old = []
    rollout_act = []
    rollout_obs_new = []
    rollout_rew = []
    obs = env.reset()
    while not rollout_done():
        obs_old = obs
        act, _ = policy.predict(obs_old)
        obs, rew, done, _ = env.step(act)

        # Track episode count.
        if end_cond == "episodes":
            episodes_elapsed += np.sum(done)

        # Don't save tuples if there is a done. The new_obs for any environment
        # is incorrect for any timestep where there is an episode end.
        # (See GH Issue #1).
        # XXX: A more efficient alternative could save the outputs from
        # environments that aren't done in this timestep. Right now we discard
        # every output in the timestep.
        if np.any(done):
            tf.logging.debug("Skipping rollout append due to done.")
            continue

        # Current state.
        rollout_obs_old.extend(obs_old)

        # Current action.
        rollout_act.extend(act)

        # Transition state and rewards.
        rollout_obs_new.extend(obs)
        rollout_rew.extend(rew)

    # Convert results to numpy arrays. (Possibly truncate).
    rollout_obs_new = np.atleast_1d(rollout_obs_new)
    rollout_obs_old = np.atleast_1d(rollout_obs_old)
    rollout_act = np.atleast_1d(rollout_act)
    rollout_rew = np.atleast_1d(rollout_rew)
    if end_cond == "timesteps" and truncate_timesteps:
        n_steps = n_timesteps

        # Truncate because we want exactly n_timesteps.
        rollout_obs_new = rollout_obs_new[:n_timesteps]
        rollout_obs_old = rollout_obs_old[:n_timesteps]
        rollout_act = rollout_act[:n_timesteps]
        rollout_rew = rollout_rew[:n_timesteps]
    else:
        n_steps = len(rollout_obs_new)

    # Sanity checks.
    exp_obs = (n_steps,) + env.observation_space.shape
    exp_act = (n_steps,) + env.action_space.shape
    assert rollout_obs_new.shape == exp_obs
    assert rollout_obs_old.shape == exp_obs
    assert rollout_act.shape == exp_act
    assert rollout_rew.shape == (n_steps,)

    return rollout_obs_old, rollout_act, rollout_obs_new, rollout_rew


def total_reward(policy, env, **kwargs):
    """
    Get the undiscounted reward after rolling out `n_timestep` steps in
    of the policy. With large n_timesteps, this can be a decent metric
    for policy performance.

    Params:
    policy (stable_baselines.BaseRLModel) -- A stable_baselines Model, trained
      on the gym environment.
    env (VecEnv or Env or str) -- The environment(s) to interact with.
    n_timesteps (int) -- The number of rewards to collect.
    n_episodes (int) -- The number of episodes to finish before we stop
      collecting rewards. Reward from parallel episodes that are underway when
      the final episode is finished is also included in the reward total.

    Return:
    total_reward (int) -- The undiscounted reward from `n_timesteps` consecutive
      actions in `env`.
    """
    _, _, _, rew = generate(policy, env, **kwargs)
    return np.sum(rew)


def generate_multiple(policies, env, n_timesteps):
    """
    Generate obs-act-obs triples from several policies. Split the desired
    number of timesteps evenly between all the policies given.

    Params:
    policies (BaseRLModel or [BaseRLModel]) -- A policy
      or a list of policies that will be used to generate
      obs-action-obs triples.

      WARNING:
      Due to the way VecEnvs handle
      episode completion states, the last obs-state-obs triple in every
      episode is omitted. (See GitHub issue #1)
    env (gym.Env) -- The environment the policy should act in.
    n_timesteps (int) -- The number of obs-action-obs
      triples to generate. If the number of policies given doesn't
      divide this number evenly, then the last policy will generate
      more timesteps.
    Returns:
    rollout_obs_old (array) -- A numpy array with shape
      `[n_timesteps] + env.observation_space.shape`. The ith observation in this
      array is the observation seen with the agent chooses action
      `rollout_act[i]`.
    rollout_act (array) -- A numpy array with shape
      `[n_timesteps] + env.action_space.shape`.
    rollout_obs_new (array) -- A numpy array with shape
      `[n_timesteps] + env.observation_space.shape`. The ith observation in this
      array is from the transition state after the agent chooses action
      `rollout_act[i]`.
    """
    try:
        policies = list(policies)
    except TypeError:
        policies = [policies]

    n_policies = len(policies)
    quot, rem = n_timesteps // n_policies, n_timesteps % n_policies
    tf.logging.debug("rollout.generate_multiple: quot={}, rem={}"
                  .format(quot, rem))

    obs_old, act, obs_new = [], [], []
    for i, pol in enumerate(policies):
        n_timesteps_ = quot
        if i == n_policies - 1:
            # The final policy also generates the remainder if
            # n_policies doesn't evenly divide n_timesteps.
            n_timesteps_ += rem

        obs_old_, act_, obs_new_, _ = generate(pol, env,
                                               n_timesteps=n_timesteps_, truncate_timesteps=True)
        obs_old.extend(obs_old_)
        act.extend(act_)
        obs_new.extend(obs_new_)

    assert len(obs_old) == len(obs_new) == len(act) == n_timesteps

    return (np.array(x) for x in (obs_old, act, obs_new))
