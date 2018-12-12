import gym

def maybe_load_env(env_or_str):
    """
    Params:
    env_or_str (str or gym.Env): The Env or its string id in Gym.

    Return:
    env (gym.Env) -- Either the original argument if it was an Env or an
      instantiated gym Env if it was a string.
    id (str) -- The environment's id.
    """
    if isinstance(env_or_str, str):
        env = gym.make(env_or_str)
    else:
        env = env_or_str
    return env


def reset_and_wrap_env_reward(env, R):
    """
    Reset the environment, and then wrap its step function so that it
    returns a custom reward based on state-action-new_state tuples.

    The old step function is saved as `env._orig_step_`.

    Param:
      env [gym.Env] -- An environment to modify in place.
      R [callable] -- The new reward function. Takes three arguments,
        `old_obs`, `action`, and `new_obs`. Returns the new reward.
        - `old_obs` is the observation made before taking the action.
        - `action` is simply the action passed to env.step().
        - `new_obs` is the observation made after taking the action. This is
          same as the observation returned by env.step().
    """
    # XXX: Look at gym wrapper class which can override step in a
    # more idiomatic way.
    old_obs = env.reset()

    # XXX: VecEnv later.
    # XXX: Consider saving a s,a pairs until the end and evaluate sim.

    orig = getattr(env, "_orig_step_", env.step)
    env._orig_step_ = orig
    def wrapped_step(action):
        nonlocal old_obs
        obs, reward, done, info = env._orig_step_(*args, **kwargs)
        wrapped_reward = R(env._old_obs_, action, obs)
        old_obs = obs
        return obs, wrapped_reward, done, info

    env.step = wrapped_step
