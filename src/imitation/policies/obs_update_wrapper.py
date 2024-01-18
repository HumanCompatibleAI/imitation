"""Updates observation for the policy to use."""

import abc
from typing import Dict, Tuple, Union

import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule

from imitation.data import wrappers as data_wrappers


class Base(ActorCriticPolicy, abc.ABC):
    """Updates the observation for the policy to use."""

    def __init__(self, policy: ActorCriticPolicy, lr_schedule: Schedule):
        """Builds the wrapper base and initializes the policy.

        Args:
            policy: The policy to wrap.
            lr_schedule: The learning rate schedule.
        """
        if policy.use_sde:
            assert policy.dist_kwargs is not None
            full_std = policy.dist_kwargs["use_sde"]
            use_expln = policy.dist_kwargs["use_expln"]
        else:
            full_std = True
            use_expln = False
        super().__init__(
            observation_space=policy.observation_space,
            action_space=policy.action_space,
            lr_schedule=lr_schedule,
            net_arch=policy.net_arch,
            activation_fn=policy.activation_fn,
            ortho_init=policy.ortho_init,
            use_sde=policy.use_sde,
            log_std_init=policy.log_std_init,
            full_std=full_std,
            use_expln=use_expln,
            share_features_extractor=policy.share_features_extractor,
            squash_output=policy.squash_output,
            features_extractor_class=policy.features_extractor_class,
            features_extractor_kwargs=policy.features_extractor_kwargs,
            normalize_images=policy.normalize_images,
            optimizer_class=policy.optimizer_class,
            optimizer_kwargs=policy.optimizer_kwargs,
        )

    @abc.abstractmethod
    def _update_ob(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Updates the observation for the policy to use."""

    def _predict(
        self,
        observation: th.Tensor,
        deterministic: bool = False,
    ) -> th.Tensor:
        """Gets the action according to the policy for a given observation."""
        return super()._predict(observation, deterministic)

    def is_vectorized_observation(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> bool:
        """Checks whether or not the observation is vectorized."""
        observation = self._update_ob(observation)
        return super().is_vectorized_observation(observation)

    def obs_to_tensor(
        self,
        observation: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> Tuple[th.Tensor, bool]:
        """Converts an observation to a PyTorch tensor that can be fed to a model."""
        observation = self._update_ob(observation)
        return super().obs_to_tensor(observation)


class RemoveHR(Base):
    """Removes human readable observation for the policy to use."""

    def __init__(self, policy: ActorCriticPolicy, lr_schedule: Schedule):
        """Builds the wrapper that removes human readable observation for the policy.

        Args:
            policy: The policy to wrap.
            lr_schedule: The learning rate schedule.
        """
        super().__init__(policy, lr_schedule)

    def _update_ob(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Removes the human readable observation if any."""
        return _remove_hr_obs(obs)


def _remove_hr_obs(
    obs: Union[np.ndarray, Dict[str, np.ndarray]],
) -> Union[np.ndarray, Dict[str, np.ndarray]]:
    """Removes the human readable observation if any."""
    if not isinstance(obs, dict):
        return obs
    if data_wrappers.HR_OBS_KEY not in obs:
        return obs
    if len(obs) == 1:
        raise ValueError(
            "Only human readable observation exists, can't remove it",
        )
    # keeps the original observation unchanged in case it is used elsewhere.
    new_obs = obs.copy()
    del new_obs[data_wrappers.HR_OBS_KEY]
    if len(new_obs) == 1:
        # unwrap dictionary structure
        return next(iter(new_obs.values()))  # type: ignore[return-value]
    return new_obs
