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
