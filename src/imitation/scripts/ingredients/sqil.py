"""This ingredient provides a sqil algorithm instance."""
import sacred

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
