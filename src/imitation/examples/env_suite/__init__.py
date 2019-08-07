from typing import Optional

from gym.envs import register as gym_register

_ENTRY_POINT_PREFIX = 'imitation.examples.env_suite'


def _register(env_name: str, entry_point: str, kwargs: Optional[dict] = None):
    entry_point = f"{_ENTRY_POINT_PREFIX}.{entry_point}"
    gym_register(id=env_name, entry_point=entry_point, kwargs=kwargs)


# hopefully will test for other sizes
for map_size in [2]:
    _register(f'imitation/Mnist{map_size}x{map_size}-v0',
              entry_point=f'mnist:MnistEnv',
              kwargs={'map_size': map_size})
