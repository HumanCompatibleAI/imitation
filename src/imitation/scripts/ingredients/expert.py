"""This ingredient provides an expert policy.

The expert policy is either loaded from disk or from the HuggingFace Model Hub or is
a test policy (e.g., random or zero).
The supported policy types are:

- `ppo` and `sac`: A policy trained with SB3. Needs a `path` in the `loader_kwargs`.
- `<algo>-huggingface` (algo can be `ppo` or `sac`): A policy trained with SB3 and uploaded to the HuggingFace Model
    Hub. Will load the model from the repo `<organization>/<algo>-<env_name>`.
    You can set the organization with the `organization` key in `loader_kwargs`. The default is `HumanCompatibleAI`.
- `random`: A policy that takes random actions.
- `zero`: A policy that takes zero actions.
"""
import sacred

from imitation.policies import serialize
from imitation.scripts.ingredients import environment

expert_ingredient = sacred.Ingredient(
    "expert",
    ingredients=[environment.environment_ingredient],
)


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
        e_config["loader_kwargs"]["env_name"] = config["environment"]["gym_id"]

    # Note: this only serves the purpose to indicated that you need to specify the
    #   path for local policies. It makes the config more explicit.
    if (
        e_config["policy_type"] in ("ppo", "sac")
        and "path" not in e_config["loader_kwargs"]
    ):  # pragma: no cover
        e_config["loader_kwargs"]["path"] = None
    return e_config


@expert_ingredient.capture
def get_expert_policy(venv, policy_type, loader_kwargs):
    return serialize.load_policy(policy_type, venv, **loader_kwargs)
