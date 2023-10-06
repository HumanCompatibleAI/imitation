"""Class to filter human readable observation for the policy to use."""

from typing import Optional, Dict, Union, Tuple

import numpy as np
import torch as th
from stable_baselines3.common.policies import BasePolicy, ActorCriticPolicy
from stable_baselines3.common.distributions import Distribution

from imitation.data import wrappers as data_wrappers


class BasePolicyWrapper(BasePolicy):
    def __init__(self, policy: BasePolicy):
        BasePolicy.__init__(
            self,
            observation_space=policy.observation_space,
            action_space=policy.action_space,
            features_extractor_class=policy.features_extractor_class,
            features_extractor_kwargs=policy.features_extractor_kwargs,
            features_extractor=policy.features_extractor,
            normalize_images=policy.normalize_images,
            optimizer_class=policy.optimizer_class,
            optimizer_kwargs=policy.optimizer_kwargs,
            squash_output=policy.squash_output,
        )
        self.policy = policy

    def predict(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
        state: Optional[Tuple[np.ndarray, ...]] = None,
        episode_start: Optional[np.ndarray] = None,
        deterministic: bool = False,
    ) -> Union[Dict[str, np.ndarray], Dict[str, th.Tensor], np.ndarray, th.Tensor]:
        obs = self._ob_filter(obs)
        return self.policy.predict(obs, state, episode_start, deterministic)

    def _ob_filter(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        return obs


class ActorCriticPolicyWrapper(BasePolicyWrapper, ActorCriticPolicy):
    def __init__(self, policy: ActorCriticPolicy):
        BasePolicyWrapper.__init__(self, policy)
        self.policy = policy

    def _tensor_ob_filter(
        self,
        obs: Union[th.Tensor, Dict[str, th.Tensor]],
    ) -> th.Tensor:
        assert isinstance(obs, th.Tensor)
        return obs

    def forward(
        self,
        obs: Union[th.Tensor, Dict[str, th.Tensor]],
        deterministic: bool = False,
    ) -> Tuple[th.Tensor, th.Tensor, th.Tensor]:
        obs = self._tensor_ob_filter(obs)
        return self.policy.forward(obs, deterministic)

    def extract_features(
        self,
        obs: Union[th.Tensor, Dict[str, th.Tensor]],
    ) -> Union[th.Tensor, Tuple[th.Tensor, th.Tensor]]:
        obs = self._tensor_ob_filter(obs)
        return self.policy.extract_features(obs)

    def evaluate_actions(
        self, obs: Union[th.Tensor, Dict[str, th.Tensor]], actions: th.Tensor
    ) -> Tuple[th.Tensor, th.Tensor, Optional[th.Tensor]]:
        obs = self._tensor_ob_filter(obs)
        return self.policy.evaluate_actions(obs, actions)

    def get_distribution(
        self,
        obs: Union[th.Tensor, Dict[str, th.Tensor]],
    ) -> Distribution:
        obs = self._tensor_ob_filter(obs)
        return self.policy.get_distribution(obs)

    def predict_values(self, obs: Union[th.Tensor, Dict[str, th.Tensor]]) -> th.Tensor:
        obs = self._tensor_ob_filter(obs)
        return self.policy.predict_values(obs)


class ExcludeHRWrapper(ActorCriticPolicyWrapper):
    """Excludes human readable observation for the policy."""

    def __init__(self, policy):
        super().__init__(policy)

    def _ob_filter(
        self,
        obs: Union[np.ndarray, Dict[str, np.ndarray]],
    ) -> Union[np.ndarray, Dict[str, np.ndarray]]:
        assert isinstance(obs, dict)
        new_obs = _remove_hr_obs(obs)
        assert isinstance(new_obs, (np.ndarray, dict))
        return new_obs

    def _tensor_ob_filter(
        self,
        obs: Union[th.Tensor, Dict[str, th.Tensor]],
    ) -> th.Tensor:
        assert isinstance(obs, dict)
        new_obs = _remove_hr_obs(obs)
        assert isinstance(new_obs, th.Tensor)
        return new_obs


def _remove_hr_obs(
    obs: Union[Dict[str, np.ndarray], Dict[str, th.Tensor]],
) -> Union[Dict[str, np.ndarray], Dict[str, th.Tensor], np.ndarray, th.Tensor]:
    """Removes human readable observation from the observation."""
    assert data_wrappers.HR_OBS_KEY in obs
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
