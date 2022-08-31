"""Common configuration elements for loading of expert policies."""
import sacred

from imitation.policies import serialize
from imitation.scripts.common.common import common_ingredient

expert_ingredient = sacred.Ingredient("expert", ingredients=[common_ingredient])


@expert_ingredient.config
def config():
    policy_type = "ppo-huggingface"
    loader_kwargs = dict()
    locals()  # quieten flake8


@expert_ingredient.capture
def get_expert_policy(venv, policy_type, loader_kwargs, common):
    if "huggingface" in policy_type:  # TODO(max): this is a hack
        loader_kwargs = loader_kwargs.copy()
        loader_kwargs["env_id"] = common["env_name"]
    return serialize.load_policy(policy_type, venv, **loader_kwargs)
