"""Single source of truth for seeding random number generators in experiments."""

from typing import Optional

import numpy as np
import sacred

seeding_ingredient = sacred.Ingredient("seeding")


@seeding_ingredient.config
def config():
    seed = 0
    locals()  # quieten flake8


@seeding_ingredient.capture
def get_seed(seed: Optional[int]) -> Optional[int]:
    """Returns the seed used by the seeding ingredient."""
    return seed


@seeding_ingredient.capture
def make_random_state(seed: Optional[int]) -> np.random.RandomState:
    """Creates a `np.random.RandomState` with the given seed."""
    return np.random.RandomState(seed)
