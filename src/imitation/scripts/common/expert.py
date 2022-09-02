"""Common configuration elements for loading of expert policies."""
import sacred

from imitation.policies import serialize
from imitation.scripts.common.common import common_ingredient

expert_ingredient = sacred.Ingredient("expert", ingredients=[common_ingredient])


@expert_ingredient.config
def config():
    # [ppo, sac, random, zero, huggingface-ppo, huggingface-sac] or your own.
    policy_type = "ppo-huggingface"
    # See imitation.policies.serialize.load_policy for options.
    loader_kwargs = dict()
    locals()  # quieten flake8


@expert_ingredient.capture
def get_expert_policy(venv, policy_type, loader_kwargs, common):
    if "huggingface" in policy_type:
        # Note: unfortunately we need to pass the venv **and** its name to the
        # huggingface policy loader since there is no way to get the name from the venv.
        # The name is needed to deduce the repo id and load the correct huggingface
        # model.
        loader_kwargs = loader_kwargs.copy()
        loader_kwargs["env_name"] = common["env_name"]
    return serialize.load_policy(policy_type, venv, **loader_kwargs)
