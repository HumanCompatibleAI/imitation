"""Load serialized policies of different types."""

# FIXME(sam): it seems like this module could mostly be replaced with a few
# torch.load() and torch.save() calls

import logging
import pathlib
from typing import Callable, Type, TypeVar

import huggingface_sb3 as hfsb3
from stable_baselines3.common import base_class, callbacks, policies, vec_env

import imitation.data.serialize
from imitation.policies import base
from imitation.util import registry

Algorithm = TypeVar("Algorithm", bound=base_class.BaseAlgorithm)

# Note: a VecEnv will always be passed first and then any kwargs. There is just no
# proper way to specify this in python yet. For details see
# https://stackoverflow.com/questions/61569324/type-annotation-for-callable-that-takes-kwargs
# TODO(juan) this can be fixed using ParamSpec
#  (https://github.com/HumanCompatibleAI/imitation/issues/574)
PolicyLoaderFn = Callable[..., policies.BasePolicy]
"""A policy loader function that takes a VecEnv before any other custom arguments and
returns a stable_baselines3 base policy policy."""

policy_registry: registry.Registry[PolicyLoaderFn] = registry.Registry()
"""Registry of policy loading functions. Add your own here if desired."""


def load_stable_baselines_model(
    cls: Type[Algorithm],
    path: str,
    venv: vec_env.VecEnv,
    **kwargs,
) -> Algorithm:
    """Helper method to load RL models from Stable Baselines.

    Args:
        cls: Stable Baselines RL algorithm.
        path: Path to zip file containing saved model data or to a folder containing a
            `model.zip` file.
        venv: Environment to train on.
        kwargs: Passed through to `cls.load`.

    Raises:
        FileNotFoundError: If `path` is not a directory containing a `model.zip` file.
        FileExistsError: If `path` contains a `vec_normalize.pkl` file (unsupported).

    Returns:
        The deserialized RL algorithm.
    """
    logging.info(f"Loading Stable Baselines policy for '{cls}' from '{path}'")
    path_obj = imitation.data.serialize.parse_path(path)

    if path_obj.is_dir():
        path_obj = path_obj / "model.zip"
        if not path_obj.exists():
            raise FileNotFoundError(
                f"Expected '{path}' to be a directory containing a 'model.zip' file.",
            )

    # SOMEDAY(adam): added 2022-01, can probably remove this check in 2023
    vec_normalize_path = path_obj.parent / "vec_normalize.pkl"
    if vec_normalize_path.exists():
        raise FileExistsError(
            "Outdated policy format: we do not support restoring normalization "
            f"statistics from '{vec_normalize_path}'",
        )

    return cls.load(path_obj, env=venv, **kwargs)


def _load_stable_baselines_from_file(
    cls: Type[base_class.BaseAlgorithm],
) -> PolicyLoaderFn:
    """Creates a policy loading function to read a policy from a file.

    Args:
        cls: The RL algorithm, e.g. `stable_baselines3.PPO`.

    Returns:
        A function loading policies trained via cls.
    """

    def f(venv: vec_env.VecEnv, path: str) -> policies.BasePolicy:
        """Loads a policy saved to path, for environment env."""
        model = load_stable_baselines_model(cls, path, venv)
        return getattr(model, "policy")

    return f


def _load_stable_baselines_from_huggingface(
    algo_name: str,
    cls: Type[base_class.BaseAlgorithm],
) -> PolicyLoaderFn:
    """Creates a policy loading function to load from Hugging Face.

    Args:
        algo_name: The name of the algorithm, e.g. `ppo`.
        cls: The RL algorithm, e.g. `stable_baselines3.PPO`.

    Returns:
        A function loading policies trained via cls.
    """

    def f(
        venv: vec_env.VecEnv,
        env_name: str,
        organization: str = "HumanCompatibleAI",
    ) -> policies.BasePolicy:
        """Loads a policy saved to path, for environment env."""
        model_name = hfsb3.ModelName(algo_name, hfsb3.EnvironmentName(env_name))
        repo_id = hfsb3.ModelRepoId(organization, model_name)
        filename = hfsb3.load_from_hub(repo_id, model_name.filename)
        model = load_stable_baselines_model(cls, filename, venv)
        return getattr(model, "policy")

    return f


policy_registry.register(
    "random",
    value=registry.build_loader_fn_require_space(base.RandomPolicy),
)
policy_registry.register(
    "zero",
    value=registry.build_loader_fn_require_space(base.ZeroPolicy),
)


def _add_stable_baselines_policies_from_file(classes):
    for k, cls_name in classes.items():
        cls = registry.load_attr(cls_name)
        fn = _load_stable_baselines_from_file(cls)
        policy_registry.register(k, value=fn)


def _add_stable_baselines_policies_from_huggingface(classes):
    for k, cls_name in classes.items():
        cls = registry.load_attr(cls_name)
        fn = _load_stable_baselines_from_huggingface(k, cls)
        policy_registry.register(f"{k}-huggingface", value=fn)


STABLE_BASELINES_CLASSES = {
    "ppo": "stable_baselines3:PPO",
    "sac": "stable_baselines3:SAC",
}
_add_stable_baselines_policies_from_file(STABLE_BASELINES_CLASSES)
_add_stable_baselines_policies_from_huggingface(STABLE_BASELINES_CLASSES)


def load_policy(
    policy_type: str,
    venv: vec_env.VecEnv,
    **kwargs,
) -> policies.BasePolicy:
    """Load serialized policy.

    Note on the kwargs:

    - `zero` and `random` policy take no kwargs
    - `ppo` and `sac` policies take a `path` argument with a path to a zip file or to a
      folder containing a `model.zip` file.
    - `ppo-huggingface` and `sac-huggingface` policies take an `env_name` and optional
      `organization` argument.

    Args:
        policy_type: A key in `policy_registry`, e.g. `ppo`.
        venv: An environment that the policy is to be used with.
        **kwargs: Additional arguments to pass to the policy loader.

    Returns:
        The deserialized policy.
    """
    agent_loader = policy_registry.get(policy_type)
    return agent_loader(venv, **kwargs)


def save_stable_model(
    output_dir: pathlib.Path,
    model: base_class.BaseAlgorithm,
    filename: str = "model.zip",
) -> None:
    """Serialize Stable Baselines model.

    Load later with `load_policy(..., policy_path=output_dir)`.

    Args:
        output_dir: Path to the save directory.
        model: The stable baselines model.
        filename: The filename of the model.
    """
    # Save each model in new directory in case we want to add metadata or other
    # information in future. (E.g. we used to save `VecNormalize` statistics here,
    # although that is no longer necessary.)
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save(output_dir / filename)
    logging.info(f"Saved policy to {output_dir}")


class SavePolicyCallback(callbacks.EventCallback):
    """Saves the policy using `save_stable_model` each time it is called.

    Should be used in conjunction with `callbacks.EveryNTimesteps`
    or another event-based trigger.
    """

    def __init__(
        self,
        policy_dir: pathlib.Path,
        *args,
        **kwargs,
    ):
        """Builds SavePolicyCallback.

        Args:
            policy_dir: Directory to save checkpoints.
            *args: Passed through to `callbacks.EventCallback`.
            **kwargs: Passed through to `callbacks.EventCallback`.
        """
        super().__init__(*args, **kwargs)
        self.policy_dir = policy_dir

    def _on_step(self) -> bool:
        assert self.model is not None
        output_dir = self.policy_dir / f"{self.num_timesteps:012d}"
        save_stable_model(output_dir, self.model)
        return True


def load_rollouts_from_huggingface(
    algo_name: str,
    env_name: str,
    organization: str = "HumanCompatibleAI",
) -> str:
    model_name = hfsb3.ModelName(algo_name, hfsb3.EnvironmentName(env_name))
    repo_id = hfsb3.ModelRepoId(organization, model_name)
    filename = hfsb3.load_from_hub(repo_id, "rollouts.npz")
    return filename
