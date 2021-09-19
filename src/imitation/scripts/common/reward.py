"""Common configuration elements for reward network training."""

import logging

import sacred

from imitation.scripts.common.train import train_ingredient

reward_ingredient = sacred.Ingredient("reward", ingredients=[train_ingredient])
logger = logging.getLogger(__name__)
