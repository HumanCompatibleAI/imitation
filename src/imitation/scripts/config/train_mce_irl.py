"""Configuration for imitation.scripts.train_mce_irl."""
import sacred
from torch import nn
import torch as th

from imitation.scripts.ingredients import environment
from imitation.scripts.ingredients import logging as logging_ingredient
from imitation.scripts.ingredients import policy_evaluation, reward, rl

train_mce_irl_ex = sacred.Experiment(
    "train_mce_irl",
    ingredients=[
        logging_ingredient.logging_ingredient,
        environment.environment_ingredient,
        reward.reward_ingredient,
        rl.rl_ingredient,
        policy_evaluation.policy_evaluation_ingredient,
    ],
)


MUJOCO_SHARED_LOCALS = dict(rl=dict(rl_kwargs=dict(ent_coef=0.1)))
ANT_SHARED_LOCALS = dict(
    total_timesteps=int(3e7),
    rl=dict(batch_size=16384),
)


@train_mce_irl_ex.config
def train_defaults():
    mceirl = {
        "discount": 1,
        "linf_eps": 0.001,
        "grad_l2_eps": 0.0001,
        "log_interval": 100,
    }
    optimizer_cls = th.optim.Adam
    optimizer_kwargs = dict(
        lr=4e-4,
    )
    env_kwargs = {
        "height": 4,
        "horizon": 40,
        "width": 7,
        "use_xy_obs": True,
    }
    num_vec = 8  # number of environments in VecEnv
    parallel = False
