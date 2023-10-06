"""Class to filter human readable observation for the policy to use."""

import abc
from typing import Dict, Union, Tuple

import numpy as np
import torch as th
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.type_aliases import Schedule

from imitation.data import wrappers as data_wrappers


class Base(ActorCriticPolicy, abc.ABC):
    def __init__(self, policy: ActorCriticPolicy, lr_schedule: Schedule):
        full_std = policy.dist_kwargs["use_sde"] if policy.use_sde else True
        use_expln = policy.dist_kwargs["use_expln"] if policy.use_sde else False
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
    def _ob_filter(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        """Filters observation for the policy to use."""

    def _predict(
        self, observation: th.Tensor, deterministic: bool = False
    ) -> th.Tensor:
        return super()._predict(observation, deterministic)

    def is_vectorized_observation(
        self, observation: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> bool:
        observation = self._ob_filter(observation)
        return super().is_vectorized_observation(observation)

    def obs_to_tensor(
        self, observation: Union[np.ndarray, Dict[str, np.ndarray]]
    ) -> Tuple[th.Tensor, bool]:
        observation = self._ob_filter(observation)
        return super().obs_to_tensor(observation)


class RemoveHR(Base):
    """Removes human readable observation for the policy."""

    def __init__(self, policy: ActorCriticPolicy, lr_schedule: Schedule):
        super().__init__(policy, lr_schedule)

    def _ob_filter(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        obs = data_wrappers.remove_hr_obs(obs)
        return obs
