"""Load serialized policies of different types."""

import contextlib
import os
import pickle
from typing import Callable, ContextManager, Iterator, Optional, Type

import tensorflow as tf
from stable_baselines.common.base_class import BaseRLModel
from stable_baselines.common.policies import BasePolicy
from stable_baselines.common.vec_env import VecEnv, VecNormalize

from imitation.policies.base import RandomPolicy, ZeroPolicy
from imitation.util import registry

PolicyLoaderFn = Callable[[str, VecEnv], ContextManager[BasePolicy]]

policy_registry: registry.Registry[PolicyLoaderFn] = registry.Registry()


class NormalizePolicy(BasePolicy):
    """Wraps a policy, normalizing its input observations.

    `VecNormalize` normalizes observations to have zero mean and unit standard
    deviation. To do this, it collects statistics on the observations. We must
    restore these statistics when we load the policy, or we will be feeding
    observations in of a different scale to those the policy was trained with.

    It is convenient to do this when loading the policy, so users of a saved
    policy are not responsible for this implementation detail. WARNING: This
    trick will not work for fine-tuning / training policies.
    """

    def __init__(self, policy: BasePolicy, vec_normalize: VecNormalize):
        super().__init__(
            policy.sess,
            policy.ob_space,
            policy.ac_space,
            policy.n_env,
            policy.n_steps,
            policy.n_batch,
        )
        self._policy = policy
        self.vec_normalize = vec_normalize

    def _wrapper(self, fn, obs, state=None, mask=None, *args, **kwargs):
        norm_obs = self.vec_normalize.normalize_obs(obs)
        return fn(norm_obs, state=state, mask=mask, *args, **kwargs)

    def step(self, *args, **kwargs):
        return self._wrapper(self._policy.step, *args, **kwargs)

    def proba_step(self, *args, **kwargs):
        return self._wrapper(self._policy.proba_step, *args, **kwargs)


def _load_stable_baselines(cls: Type[BaseRLModel], policy_attr: str) -> PolicyLoaderFn:
    """Higher-order function, returning a policy loading function.

    Args:
        cls: The RL algorithm, e.g. `stable_baselines.PPO2`.
        policy_attr: The attribute of the RL algorithm containing the policy,
            e.g. `act_model`.

    Returns:
        A function loading policies trained via cls.
    """

    @contextlib.contextmanager
    def f(path: str, venv: VecEnv) -> Iterator[BasePolicy]:
        """Loads a policy saved to path, for environment env."""
        tf.logging.info(
            f"Loading Stable Baselines policy for '{cls}' " f"from '{path}'"
        )
        model_path = os.path.join(path, "model.pkl")
        model = None
        try:
            model = cls.load(model_path, env=venv)
            policy = getattr(model, policy_attr)

            try:
                normalize_path = os.path.join(path, "vec_normalize.pkl")
                with open(normalize_path, "rb") as f:
                    vec_normalize = pickle.load(f)
                vec_normalize.training = False
                vec_normalize.set_venv(venv)
                policy = NormalizePolicy(policy, vec_normalize)
                tf.logging.info(f"Loaded VecNormalize from '{normalize_path}'")
            except FileNotFoundError:
                # We did not use VecNormalize during training, skip
                pass

            yield policy
        finally:
            if model is not None and model.sess is not None:
                model.sess.close()

    return f


policy_registry.register(
    "random",
    value=registry.build_loader_fn_require_space(registry.dummy_context(RandomPolicy),),
)
policy_registry.register(
    "zero",
    value=registry.build_loader_fn_require_space(registry.dummy_context(ZeroPolicy),),
)


def _add_stable_baselines_policies(classes):
    for k, (cls_name, attr) in classes.items():
        try:
            cls = registry.load_attr(cls_name)
            fn = _load_stable_baselines(cls, attr)
            policy_registry.register(k, value=fn)
        except (AttributeError, ImportError):
            # We expect PPO1 load to fail if mpi4py isn't installed.
            # Stable Baselines can be installed without mpi4py.
            tf.logging.debug(f"Couldn't load {cls_name}. Skipping...")


STABLE_BASELINES_CLASSES = {
    "ppo1": ("stable_baselines:PPO1", "policy_pi"),
    "ppo2": ("stable_baselines:PPO2", "act_model"),
}
_add_stable_baselines_policies(STABLE_BASELINES_CLASSES)


def load_policy(
    policy_type: str, policy_path: str, venv: VecEnv
) -> ContextManager[BasePolicy]:
    """Load serialized policy.

    Args:
        policy_type: A key in `policy_registry`, e.g. `ppo2`.
        policy_path: A path on disk where the policy is stored.
        venv: An environment that the policy is to be used with.
    """
    agent_loader = policy_registry.get(policy_type)
    return agent_loader(policy_path, venv)


def save_stable_model(
    output_dir: str, model: BaseRLModel, vec_normalize: Optional[VecNormalize] = None,
) -> None:
    """Serialize policy.

    Load later with `load_policy(..., policy_path=output_dir)`.

    Args:
        output_dir: Path to the save directory.
        policy: The stable baselines policy.
        vec_normalize: Optionally, a VecNormalize to save statistics for.
            `load_policy` automatically applies `NormalizePolicy` wrapper
            when loading.
    """
    os.makedirs(output_dir, exist_ok=True)
    model.save(os.path.join(output_dir, "model.pkl"))
    if vec_normalize is not None:
        with open(os.path.join(output_dir, "vec_normalize.pkl"), "wb") as f:
            pickle.dump(vec_normalize, f)
    tf.logging.info("Saved policy to %s", output_dir)
