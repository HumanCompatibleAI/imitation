"""This ingredient provides a SQIL algorithm instance."""
import sacred
from stable_baselines3 import dqn as dqn_algorithm

from imitation.policies import base
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

@rl.rl_ingredient.config_hook
def config_hook(config, command_name, logger):
    # want to remove arguments added by rl but keep the ones that are added by others
    del logger

    res = {}
    if command_name == "sqil" and config["rl"]["rl_cls"] is None:
        res["rl_cls"] = dqn_algorithm.DQN
        res["rl_kwargs"] = {}
    
    return res

@policy.policy_ingredient.config_hook
def config_hook(config, command_name, logger):
    del logger

    res = {}
    if command_name == "sqil" and config["policy"]["policy_cls"] == base.FeedForward32Policy:
        res["policy_cls"] = "MlpPolicy"
    
    return res
