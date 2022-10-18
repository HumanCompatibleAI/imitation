"""Common configuration elements for loading of expert policies."""
import sacred

from imitation.policies import serialize
from imitation.scripts.common.common import common_ingredient

expert_ingredient = sacred.Ingredient("expert", ingredients=[common_ingredient])


@expert_ingredient.config
def config():
    # [ppo, sac, random, zero, ppo-huggingface, sac-huggingface] or your own.
    policy_type = "ppo-huggingface"
    # See imitation.policies.serialize.load_policy for options.
    loader_kwargs = dict()
    locals()  # quieten flake8


@expert_ingredient.config_hook
def config_hook(config, command_name, logger):
    e_config = config["expert"]
    if "huggingface" in e_config["policy_type"]:
        # Set the default loader_kwargs for huggingface policies.
        if "organization" not in e_config["loader_kwargs"]:
            e_config["loader_kwargs"]["organization"] = "HumanCompatibleAI"

        # Note: unfortunately we need to pass the venv **and** its name to the
        # huggingface policy loader since there is no way to get the name from the venv.
        # The name is needed to deduce the repo id and load the correct huggingface
        # model.
        e_config["loader_kwargs"]["env_name"] = config["common"]["env_name"]

    # Note: this only serves the purpose to indicated that you need to specify the
    #   path for local policies. It makes the config more explicit.
    if (
        e_config["policy_type"] in ("ppo", "sac")
        and "path" not in e_config["loader_kwargs"]
    ):  # pragma: no cover
        e_config["loader_kwargs"]["path"] = None
    return e_config


@expert_ingredient.capture
def get_expert_policy(venv, policy_type, loader_kwargs, common):
    return serialize.load_policy(policy_type, venv, **loader_kwargs)
