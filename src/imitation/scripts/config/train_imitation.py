"""Configuration settings for train_dagger, training DAgger from synthetic demos."""

import sacred
from stable_baselines3 import dqn

from imitation.scripts.ingredients import bc
from imitation.scripts.ingredients import demonstrations as demos_common
from imitation.scripts.ingredients import environment, expert
from imitation.scripts.ingredients import logging as logging_ingredient
from imitation.scripts.ingredients import policy_evaluation

train_imitation_ex = sacred.Experiment(
    "train_imitation",
    ingredients=[
        logging_ingredient.logging_ingredient,
        demos_common.demonstrations_ingredient,
        expert.expert_ingredient,
        environment.environment_ingredient,
        policy_evaluation.policy_evaluation_ingredient,
        bc.bc_ingredient,
    ],
)


@train_imitation_ex.config
def config():
    dagger = dict(
        use_offline_rollouts=False,  # warm-start policy with BC from offline demos
        total_timesteps=1e5,
        beta_schedule=None,
    )

    sqil = dict(
        rl_algo_class=dqn.DQN,
        rl_kwargs=dict(),
        total_timesteps=3e5,
        policy_model="MlpPolicy",  # TODO: this is a placeholder, change it to something that makes sense
        train_kwargs=dict(
            log_interval=4,  # Number of updates between Tensorboard/stdout logs
            progress_bar=True,
        ),
    )


@train_imitation_ex.named_config
def mountain_car():
    environment = dict(gym_id="MountainCar-v0")
    bc = dict(l2_weight=0.0)
    dagger = dict(total_timesteps=20000)


@train_imitation_ex.named_config
def seals_mountain_car():
    environment = dict(gym_id="seals/MountainCar-v0")
    bc = dict(l2_weight=0.0)
    dagger = dict(total_timesteps=20000)


@train_imitation_ex.named_config
def cartpole():
    environment = dict(gym_id="CartPole-v1")
    dagger = dict(total_timesteps=20000)


@train_imitation_ex.named_config
def seals_cartpole():
    environment = dict(gym_id="seals/CartPole-v0")
    dagger = dict(total_timesteps=20000)


@train_imitation_ex.named_config
def pendulum():
    environment = dict(gym_id="Pendulum-v1")


@train_imitation_ex.named_config
def ant():
    environment = dict(gym_id="Ant-v2")


@train_imitation_ex.named_config
def seals_ant():
    environment = dict(gym_id="seals/Ant-v0")


@train_imitation_ex.named_config
def half_cheetah():
    environment = dict(gym_id="HalfCheetah-v2")
    bc = dict(l2_weight=0.0)
    dagger = dict(total_timesteps=60000)


@train_imitation_ex.named_config
def seals_half_cheetah():
    environment = dict(gym_id="seals/HalfCheetah-v0")
    bc = dict(l2_weight=0.0)
    dagger = dict(total_timesteps=60000)


@train_imitation_ex.named_config
def humanoid():
    environment = dict(gym_id="Humanoid-v2")


@train_imitation_ex.named_config
def seals_humanoid():
    environment = dict(gym_id="seals/Humanoid-v0")


@train_imitation_ex.named_config
def fast():
    dagger = dict(total_timesteps=50)
    bc = dict(train_kwargs=dict(n_batches=50))
    sqil = dict(total_timesteps=50)
