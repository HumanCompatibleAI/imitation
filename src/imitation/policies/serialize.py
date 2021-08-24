"""Load serialized policies of different types."""

# FIXME(sam): it seems like this module could mostly be replaced with a few
# torch.load() and torch.save() calls

import logging
import os
import pathlib
import pickle
from typing import Callable, Optional, Tuple, Type, Union

import numpy as np
import torch as th
from stable_baselines3.common import callbacks, on_policy_algorithm, policies, vec_env
from torch import nn

from imitation.policies import base
from imitation.util import registry

PolicyLoaderFn = Callable[[str, vec_env.VecEnv], policies.BasePolicy]

policy_registry: registry.Registry[PolicyLoaderFn] = registry.Registry()


class NormalizePolicy(policies.BasePolicy):
    """Wraps a policy, normalizing its input observations.

    `VecNormalize` normalizes observations to have zero mean and unit standard
    deviation. To do this, it collects statistics on the observations. We must
    restore these statistics when we load the policy, or we will be feeding
    observations in of a different scale to those the policy was trained with.

    It is convenient to do this when loading the policy, so users of a saved
    policy are not responsible for this implementation detail. WARNING: This
    trick will not work for fine-tuning / training policies.
    """

    def __init__(
        self, policy: policies.BasePolicy, vec_normalize: vec_env.VecNormalize
    ):
        super().__init__(
            observation_space=policy.observation_space,
            action_space=policy.action_space,
        )
        self._policy = policy
        self.vec_normalize = vec_normalize

    def _predict(self, *args, **kwargs):
        raise NotImplementedError()

    def predict(
        self, obs: np.ndarray, *args, **kwargs
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        preproc_obs = self.vec_normalize.normalize_obs(obs)
        return self._policy.predict(preproc_obs, *args, **kwargs)

    # next few methods are meant to prevent users from accidentally treating
    # this as a real policy

    def forward(*args, **kwargs):
        raise NotImplementedError()

    @property
    def squash_output(self) -> bool:
        raise NotImplementedError()

    @staticmethod
    def init_weights(module: nn.Module, gain: float = 1) -> None:
        raise NotImplementedError()

    def scale_action(self, action: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def unscale_action(self, scaled_action: np.ndarray) -> np.ndarray:
        raise NotImplementedError()

    def save(self, path: str) -> None:
        raise NotImplementedError()

    @classmethod
    def load(
        cls, path: str, device: Union[th.device, str] = "auto"
    ) -> policies.BasePolicy:
        raise NotImplementedError()

    def load_from_vector(self, vector: np.ndarray):
        raise NotImplementedError()

    def parameters_to_vector(self) -> np.ndarray:
        raise NotImplementedError()


def _load_stable_baselines(
    cls: Type[on_policy_algorithm.OnPolicyAlgorithm], policy_attr: str
) -> PolicyLoaderFn:
    """Higher-order function, returning a policy loading function.

    Args:
        cls: The RL algorithm, e.g. `stable_baselines3.PPO`.
        policy_attr: The attribute of the RL algorithm containing the policy,
            e.g. `act_model`.

    Returns:
        A function loading policies trained via cls.
    """

    def f(path: str, venv: vec_env.VecEnv) -> policies.BasePolicy:
        """Loads a policy saved to path, for environment env."""
        logging.info(f"Loading Stable Baselines policy for '{cls}' from '{path}'")
        policy_dir = pathlib.Path(path)
        if not policy_dir.is_dir():
            raise FileNotFoundError(
                f"path={path} needs to be a directory containing model.zip and "
                "optionally vec_normalize.pkl."
            )

        model_path = policy_dir / "model.zip"
        if not model_path.is_file():
            # Couldn't find model.zip. Try deprecated model.pkl instead?
            deprecated_model_path = policy_dir / "model.pkl"
            if deprecated_model_path.is_file():
                import warnings

                warnings.warn(
                    "Using deprecated policy directory containing model.pkl "
                    "instead of model.zip (in either case, SB3 actually saves a ZIP"
                    "file, not a .pkl file). A future version of imitation will not be "
                    "compatible with `model.pkl`. You can fix this warning now by "
                    "renaming the ZIP file: \n"
                    f"mv '{deprecated_model_path}' '{model_path}'",
                    DeprecationWarning,
                )
                model_path = deprecated_model_path
            else:
                raise FileNotFoundError(
                    f"Could not find {model_path} or (deprecated) "
                    f"{deprecated_model_path}"
                )

        model = cls.load(model_path, env=venv)
        policy = getattr(model, policy_attr)

        normalize_path = os.path.join(path, "vec_normalize.pkl")
        try:
            with open(normalize_path, "rb") as f:
                vec_normalize = pickle.load(f)
        except FileNotFoundError:
            # We did not use VecNormalize during training, skip
            pass
        else:
            vec_normalize.training = False
            vec_normalize.set_venv(venv)
            policy = NormalizePolicy(policy, vec_normalize)
            logging.info(f"Loaded VecNormalize from '{normalize_path}'")

        return policy

    return f


policy_registry.register(
    "random",
    value=registry.build_loader_fn_require_space(base.RandomPolicy),
)
policy_registry.register(
    "zero",
    value=registry.build_loader_fn_require_space(base.ZeroPolicy),
)


def _add_stable_baselines_policies(classes):
    for k, (cls_name, attr) in classes.items():
        cls = registry.load_attr(cls_name)
        fn = _load_stable_baselines(cls, attr)
        policy_registry.register(k, value=fn)


# TODO(shwang): For all subclasses of stable_baselines3.common.base_class.BaseAlgorithm,
#  the policy is saved as `self.policy`. So the second part of this mapping at least
#  might not be necessary?
STABLE_BASELINES_CLASSES = {
    "ppo": ("stable_baselines3:PPO", "policy"),
}
_add_stable_baselines_policies(STABLE_BASELINES_CLASSES)


def load_policy(
    policy_type: str, policy_path: str, venv: vec_env.VecEnv
) -> policies.BasePolicy:
    """Load serialized policy.

    Args:
        policy_type: A key in `policy_registry`, e.g. `ppo`.
        policy_path: A path on disk where the policy is stored.
        venv: An environment that the policy is to be used with.
    """
    agent_loader = policy_registry.get(policy_type)
    return agent_loader(policy_path, venv)


def save_stable_model(
    output_dir: str,
    model: on_policy_algorithm.OnPolicyAlgorithm,
    vec_normalize: Optional[vec_env.VecNormalize] = None,
) -> None:
    """Serialize Stable Baselines model.

    Load later with `load_policy(..., policy_path=output_dir)`.

    Args:
        output_dir: Path to the save directory.
        model: The stable baselines model.
        vec_normalize: Optionally, a VecNormalize to save statistics for.
            `load_policy` automatically applies `NormalizePolicy` wrapper
            when loading.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, "model.zip"))
    if vec_normalize is not None:
        with open(os.path.join(output_dir, "vec_normalize.pkl"), "wb") as f:
            pickle.dump(vec_normalize, f)
    logging.info("Saved policy to %s", output_dir)


class SavePolicyCallback(callbacks.EventCallback):
    """Saves the policy using `save_stable_model` each time it is called.

    Should be used in conjunction with `callbacks.EveryNTimesteps`
    or another event-based trigger.
    """

    def __init__(
        self,
        policy_dir: str,
        vec_normalize: Optional[vec_env.VecNormalize],
        *args,
        **kwargs,
    ):
        """Builds SavePolicyCallback.

        Args:
            policy_dir: Directory to save checkpoints.
            vec_normalize: If specified, VecNormalize object to save alongside policy.
        """
        super().__init__(*args, **kwargs)
        self.policy_dir = policy_dir
        self.vec_normalize = vec_normalize

    def _on_step(self) -> bool:
        output_dir = os.path.join(self.policy_dir, f"{self.num_timesteps:012d}")
        save_stable_model(output_dir, self.model, self.vec_normalize)
