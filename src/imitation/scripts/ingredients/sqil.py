"""This ingredient provides a sqil algorithm instance."""
import sacred
from stable_baselines3 import dqn as dqn_algorithm

from imitation.scripts.ingredients import policy, rl

sqil_ingredient = sacred.Ingredient(
    "sqil",
    ingredients=[rl.rl_ingredient, policy.policy_ingredient],
)


@sqil_ingredient.config
def config():
    total_timesteps = 3e5
    train_kwargs = dict(
        log_interval=4,  # Number of updates between Tensorboard/stdout logs
        progress_bar=True,
    )

    locals()  # quieten flake8 unused variable warning


@sqil_ingredient.named_config
def dqn():
    rl.rl_ingredient.add_config(
        dict(
            rl_cls=dqn_algorithm.DQN,
        ),
    )
    policy.policy_ingredient.add_config(
        dict(
            policy_cls="MlpPolicy",
        ),
    )
