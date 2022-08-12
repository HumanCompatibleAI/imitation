"""Common configuration elements for loading of expert policies."""
import sacred
import huggingface_sb3 as hfsb3
import stable_baselines3 as sb3
from stable_baselines3.common import policies

from imitation.policies import base
from imitation.scripts.common.common import common_ingredient, make_venv

expert_ingredient = sacred.Ingredient("expert", ingredients=[common_ingredient])


@expert_ingredient.config
def config():
    policy_type = "ppo"  # right now only ppo supported
    policy_path = None  # path to zip-file containing an expert policy
    huggingface_repo_id = None
    huggingface_orga = "HumanCompatibleAI"

    locals()  # quieten flake8


@expert_ingredient.capture
def get_huggingface_repo_id(common, huggingface_repo_id, policy_type, huggingface_orga):
    if huggingface_repo_id is not None:
        return huggingface_repo_id
    else:
        return hfsb3.ModelRepoId(
            huggingface_orga,
            hfsb3.ModelName(policy_type, hfsb3.EnvironmentName(common["env_name"])),
        )


@expert_ingredient.capture
def get_expert_path(common, policy_path, policy_type):
    if policy_path is not None:
        return policy_path
    else:
        model_name = hfsb3.ModelName(policy_type, hfsb3.EnvironmentName(common["env_name"]))
        return hfsb3.load_from_hub(get_huggingface_repo_id(), model_name.filename)


@expert_ingredient.capture
def get_expert_policy(policy_type) -> policies.BasePolicy:
    env = make_venv()
    if policy_type == "ppo":
        return sb3.PPO.load(get_expert_path(), env).policy
    elif policy_type == "random":
        base.RandomPolicy(env.observation_space, env.action_space)
    elif policy_type == "zero":
        base.ZeroPolicy(env.observation_space, env.action_space)
