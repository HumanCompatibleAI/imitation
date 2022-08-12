# =============================================================
# Contains code from Dynamic Aware Reward Distance code repo
# specifically from the following files
# rewards/evaluations/model_collection.py
# rewards/evaluations/reward_collection.py

# =============================================================
from __future__ import annotations

import abc
from collections.abc import MutableMapping
from typing import Dict, Optional

import gym
import torch

from imitation.rewards.reward_distance.reward_models import RewardModel

class RewardCollection(MutableMapping):
    """A collection of reward tensors.

    This exists to encapsulate the details of dealing with multiple models at the same time,
    as well as to handle some of the reward operations.

    Args:
        label_to_rewards: A dict mapping labels to reward tensors. If none this class is empty initialized.
    """
    def __init__(self, label_to_rewards: Optional[Dict[str, torch.Tensor]] = None):
        # Using a defaultdict would simplify the implementation below,
        # but make this class more bug prone so let's keep it classic.
        self.label_to_rewards = dict()
        if label_to_rewards is not None:
            self.label_to_rewards.update(label_to_rewards)

    def __getitem__(self, key):
        return self.label_to_rewards[key]

    def __setitem__(self, key, value):
        self.label_to_rewards[key] = value

    def __delitem__(self, key):
        del self.label_to_rewards[key]

    def __iter__(self):
        return iter(self.label_to_rewards)

    def __len__(self):
        return len(self.label_to_rewards)

    def __str__(self):
        keys_string = " ".join(self.keys())
        return f"RewardCollection for labels: {keys_string}"

    def is_valid(self) -> bool:
        """Returns True if this is a valid reward collection."""
        if len(self) == 0:
            return True

        keys = list(self.keys())
        first_reward = self[keys[0]]

        if first_reward.ndim == 0:
            # Check the single reward case.
            for key in keys:
                if self[key].ndim != 0:
                    return False
        else:
            # Check the 1d tensor reward case.
            expected_reward_length = len(self[keys[0]])
            for key in keys:
                if len(self[key]) != expected_reward_length:
                    return False

        return True

    def append(self, other: RewardCollection) -> None:
        """Appends the other collection of rewards to this one.

        Args:
            other: The other collection to append.
                Must have an identical set of keys as this reward collection.
        """
        assert other.is_valid()
        assert len(self.keys()) == 0 or self.keys() == other.keys(
        ), "RewardCollection objects should only be combined with identical keys"
        for label, new_rewards in other.items():
            assert new_rewards.ndim == 1, "Rewards should be a 1d tensor."
            if label in self:
                cur_rewards = self.label_to_rewards[label]
                self.label_to_rewards[label] = torch.cat((cur_rewards, new_rewards))
            else:
                self.label_to_rewards[label] = new_rewards
        return self

    def mean(self, *args, **kwargs) -> RewardCollection:
        """Computes the mean of each labeled reward tensor and returns the resulting collection.
        Args:
            args: Arguments to foward to the individual `mean` calls.
            kwargs: Key word arguments to forward to the individual `mean` calls.
        """
        means = RewardCollection()
        for label, rewards in self.items():
            means[label] = rewards.mean(*args, **kwargs)
        assert means.is_valid()
        return means

    def __mul__(self, values: torch.Tensor) -> RewardCollection:
        """Multiplies each reward tensor by the provided values.

        This implements elementwise multiplication. Note that this isn't between two
        RewardCollections, the same value gets multiplied by each internal reward.

        Args:
            values: Tensor of values by which to multiply.

        Returns:
            A new RewardCollection with the multipled rewards.
        """
        mult_rewards = RewardCollection()
        for label, rewards in self.items():
            mult_rewards[label] = rewards * values
        assert mult_rewards.is_valid()
        return mult_rewards

    def __rmul__(self, values: torch.Tensor) -> RewardCollection:
        """Reverse multication by tensor values.

        See `__mul__`.
        """
        return self.__mul__(values)

    def __add__(self, other: RewardCollection) -> RewardCollection:
        """Adds the other reward collection to this collection.
        Args:
            other: The other reward collection to add.
        Returns:
            A reward collection where the rewards for corresponding labels in this and other are added.
        """
        assert self.keys() == other.keys()
        add_rewards = RewardCollection()
        for label, rewards in self.items():
            add_rewards[label] = rewards + other[label]
        assert add_rewards.is_valid()
        return add_rewards

    def __sub__(self, other: RewardCollection) -> RewardCollection:
        """Subtracts the other reward collection from this collection.
        Args:
            other: The other reward collection to subtract.
        Returns:
            A reward collection where the rewards for corresponding labels in this and other are subtracted.
        """
        assert self.keys() == other.keys()
        add_rewards = RewardCollection()
        for label, rewards in self.items():
            add_rewards[label] = rewards - other[label]
        assert add_rewards.is_valid()
        return add_rewards

    def reshape(self, *args) -> RewardCollection:
        """Reshapes all the rewards in the collection, passing in the provided args.

        Args:
            args: Arguments to foward to the individual `reshape` calls.
        """
        reshaped_rewards = RewardCollection()
        for label, rewards in self.items():
            reshaped_rewards[label] = rewards.reshape(*args)
        assert reshaped_rewards.is_valid()
        return reshaped_rewards

    def to(self, *args) -> RewardCollection:
        """Calls the pytorch `to` function on each reward tensor with the provided arguments.

        Args:
            args: Arguments to foward to the individual `to` calls.
        """
        # TODO(redacted): Refactor this, reshape, and other methods to a general "apply func with args" form.
        at_rewards = RewardCollection()
        for label, rewards in self.items():
            at_rewards[label] = rewards.to(*args)
        assert at_rewards.is_valid()
        return at_rewards

    @property
    def dtype(self) -> type:
        """Gets the data type of rewards stored in this collection.

        Asserts this collection is not empty, and assumes that all rewards have the same dtype.
        """
        assert len(self) > 0
        example_rewards = next(iter(self.values()))
        return example_rewards.dtype

    @property
    def device(self) -> torch.device:
        """Gets the device of rewards stored in this collection.

        Asserts this collection is not empty, and assumes that all rewards have the same device.
        """
        assert len(self) > 0
        example_rewards = next(iter(self.values()))
        return example_rewards.device

class ModelCollection: 
    """A collection of labeled models.

    Args:
        label_to_model: Dictionary mapping labels to corresponding models.
    """
    def __init__(self, label_to_model: Dict[str, RewardModel]):
        assert len(label_to_model) > 0
        self.label_to_model = label_to_model

    def rewards(
            self,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: Optional[torch.Tensor],
            terminals: Optional[torch.Tensor],
    ) -> RewardCollection:
        """Computes the rewards for each model and returns a reward collection.

        See `RewardModel.reward` for details on args.
        """
        assert len(states) == len(actions)
        assert next_states is None or len(next_states) == len(states)
        assert terminals is None or len(terminals) == len(states)

        rewards = RewardCollection()
        for label, model in self.label_to_model.items():
            # This assumes that rewards should always be flat.
            rewards[label] = model.reward(states, actions, next_states, terminals).reshape(-1)
        assert rewards.is_valid()
        return rewards

    def get_model(self, label: str) -> RewardModel:
        """Gets the model associated with the label.

        Args:
            label: The label of the model to return. Asserts in the model collection.

        Returns:
            The model with the associated label.
        """
        assert label in self.label_to_model, f"No model with label {label}"
        return self.label_to_model[label]

    def __repr__(self):
        return f"RewardCollection({self.label_to_model})"

