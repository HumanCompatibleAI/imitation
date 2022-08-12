import numpy as np
import torch

from imitation.rewards.reward_distance.collections import ModelCollection, RewardCollection
from imitation.rewards.reward_distance.transition_sampler import TransitionSampler


def compute_canonical_reward_scale(rewards: torch.Tensor, eps: float = 1e-8) -> float:
    """Computes the scaling factor to normalize the scale of the rewards."""
    assert rewards.ndim == 1
    norm = torch.norm(rewards)
    return 0 if norm < eps else 1 / norm


def compute_scale_normalized_rewards(rewards: RewardCollection, eps: float = 1e-8) -> RewardCollection:
    """Normalizes the scale of the rewards in the provided collected."""
    normalized_rewards = RewardCollection()
    for label, label_rewards in rewards.items():
        normalized_rewards[label] = label_rewards * compute_canonical_reward_scale(label_rewards, eps=eps)
    return normalized_rewards


class EPIC:
    """Computes the EPIC distance between reward functions.

    EPIC is a pseudometric with which to compare reward functions, and was proposed in the following paper:

    Gleave, Adam, Michael Dennis, Shane Legg, Stuart Russell, and Jan Leike.
    "Quantifying differences in reward functions."
    arXiv preprint arXiv:2006.13900 (2020).

    Args:
        batch_size: The size of batches to use in the mean reward calculation.
        skip_canonicalization: If True, computes rewards, but does not canonicalize them.
    """
    def __init__(self, batch_size: int = 10000, skip_canonicalization: bool = False):
        self.batch_size = batch_size
        self.skip_canonicalization = skip_canonicalization

    def compute_canonical_rewards(
            self,
            models: ModelCollection,
            states: torch.Tensor,
            actions: torch.Tensor,
            next_states: torch.Tensor,
            terminals: torch.Tensor,
            transition_sampler: TransitionSampler,
            discount: float,
            total_mean_mode: str = "per_batch_approximation",
            should_normalize_scale: bool = False,
    ) -> RewardCollection:
        """Computes a canonical representation of rewards from the provided models on the provided transitions.

        Note: If using this function in batches (i.e., calling it repeatedly with batched data), set both
        `should_subtract_total_mean` and `should_normalize_scale` to False. This functionality does not work
        in the batched case and will yield invalid results!

        Args:
            models: Collection of models with which to compute rewards.
            states: The initial states from which transitions occurred.
            actions: Actions taken in the original transitions.
            next_states: The transitioned-to states in the transitions.
            terminals: Terminal indicators for the transitions.
            transition_sampler: object that samples actions and next states for computing mean reward values.
            discount: Discount factor to use in calculation.
            total_mean_mode: The mode for how the total mean reward is calculated / applied.
            should_normalize_scale: If True, normalizes the scale of the rewards by diving by their norm.

        Returns:
            A collection of canonicalized rewards, one for each model provided.
        """
        # Assumes all transition tensors are 2D.
        assert states.ndim == 2
        assert actions.ndim == 2
        assert next_states.ndim == 2
        raw_rewards = models.rewards(states, actions, next_states, terminals)
        if self.skip_canonicalization:
            return raw_rewards

        from_state_mean_rewards = self.compute_mean_rewards(models, states, transition_sampler)
        next_state_mean_rewards = self.compute_mean_rewards(models, next_states, transition_sampler)
        total_mean_rewards = self.compute_total_mean_rewards(
            models,
            states,
            next_states,
            transition_sampler,
            from_state_mean_rewards,
            next_state_mean_rewards,
            total_mean_mode,
        )

        canonical_rewards = raw_rewards \
            + discount * next_state_mean_rewards \
            - from_state_mean_rewards \
            - discount * total_mean_rewards

        if should_normalize_scale:
            compute_scale_normalized_rewards(canonical_rewards)

        return canonical_rewards

    def compute_mean_rewards(
            self,
            models: ModelCollection,
            states: torch.Tensor,
            transition_sampler: TransitionSampler,
    ) -> torch.Tensor:
        """Computes the mean rewards for the provided states.

        Args:
            models: The collection of models with respect to which the mean reward is defined / computed.
            states: The initial states from which transitions occur, and for which the mean reward is computed.
            transition_sampler: Samples the actions and next states used in computing the mean rewards.

        Returns:
            A collection of rewards containing mean reward values for the provided states.
        """
        mean_rewards = RewardCollection()
        indices = self.compute_batch_indices(states, transition_sampler)
        for start, end in zip(indices, indices[1:]):
            cur_states = states[start:end]
            batch_size = len(cur_states)
            # Note that we want to "repeat_interleave" cur states as opposed to "repeating" them.
            # `repeat_interleave` [1,2,3] 3 times yields [1,1,1,2,2,2,3,3,3]. Since the actions,
            # next_states, and weights are of shape (batch_size, num_transitions_per_state), we
            # want to get each state num_transitions_per_state times in a row to properly
            # align with the sampled transitions (which we will flatten).
            repeated_cur_states = cur_states.repeat_interleave(transition_sampler.num_transitions_per_state, 0)

            # Sample actions, next states, and weights and flatten the first two dimensions.
            actions, next_states, weights = transition_sampler.sample(cur_states)
            actions = actions.flatten(0, 1)
            next_states = next_states.flatten(0, 1)
            weights = weights.reshape(-1)

            # Compute the weighted mean across the rewards from the sampled actions and next states.
            rewards = models.rewards(repeated_cur_states, actions, next_states, terminals=None)
            rewards = rewards * weights
            rewards = rewards.reshape(batch_size, transition_sampler.num_transitions_per_state)
            rewards = rewards.mean(dim=-1)
            assert rewards.is_valid()

            mean_rewards.append(rewards)
        return mean_rewards

    def compute_total_mean_rewards(
        self,
        models: ModelCollection,
        states: torch.Tensor,
        next_states: torch.Tensor,
        transition_sampler: TransitionSampler,
        from_state_mean_rewards: RewardCollection,
        next_state_mean_rewards: RewardCollection,
            mode: str,
    ) -> RewardCollection:
        """Computes the total mean reward according to the provided mode.

        Args:
            models: Collection of models with which to compute rewards.
            states: The initial states from which transitions occurred.
            next_states: The transitioned-to states in the transitions.
            transition_sampler: object that samples actions and next states for computing mean reward values.
            total_mean_mode: The mode for how the total mean reward is calculated / applied.
            from_state_mean_rewards: The mean reward from the original state.
            next_state_mean_rewards: The mean rewards from the next state.
            mode: The mode according to which the total mean rewards should be computed.

        Returns:
            A reward collection of total mean rewards.
        """
        # Extract some information needed by different modes.
        dtype = from_state_mean_rewards.dtype
        device = from_state_mean_rewards.device
        labels = from_state_mean_rewards.keys()

        if mode == "none":
            # Creating a collection of zero-valued rewards allows simplifies the logic above while not impacting the results.
            total_mean_rewards = RewardCollection(
                {label: torch.tensor(0, dtype=dtype, device=device)
                 for label in labels})
        elif mode == "per_batch_approximation":
            # Copying the approach taken in the original EPIC repo, compute the total mean by
            # averaging over all the other mean rewards.
            # See https://github.com/HumanCompatibleAI/evaluating-rewards/blob/master/src/evaluating_rewards/distances/epic_sample.py
            total_mean_rewards = RewardCollection().append(from_state_mean_rewards).append(
                next_state_mean_rewards).mean()
        elif mode == "conditional_per_state_in_distribution":
            total_mean_rewards = self._compute_conditional_per_state_in_distribution_total_mean_rewards(
                models,
                states,
                transition_sampler,
            )
        elif mode == "conditional_per_state_out_of_distribution":
            total_mean_rewards = self._compute_conditional_per_state_out_of_distribution_total_mean_rewards(
                models,
                states,
                next_states,
                transition_sampler,
            )
        elif mode == "conditional_per_state_out_of_distribution_quick":
            total_mean_rewards = self._compute_conditional_per_state_out_of_distribution_total_mean_rewards_quick(
                models,
                states,
                next_states,
                transition_sampler,
            )
        else:
            raise ValueError(f"Invalid total_mean_mode: {mode}")
        assert total_mean_rewards.is_valid()
        return total_mean_rewards

    def compute_batch_indices(self, states: torch.Tensor, transition_sampler: TransitionSampler) -> np.ndarray:
        """Computes the indices to use for batched calculation of rewards

        Args:
            states: The states to compute batch indices for.
            transition_sampler: The sampler that will produce some elemnts of the batch.

        Returns:
            A numpy array of indices at which to start / end each batch.
        """
        # TODO(redacted): Base this on the amount of memory
        # available if it runs out of memory or is slow and it matters.
        del transition_sampler
        indices = np.arange(0, len(states) + self.batch_size, self.batch_size)
        return indices

    def _compute_conditional_per_state_in_distribution_total_mean_rewards(
            self,
            models: ModelCollection,
            states: torch.Tensor,
            transition_sampler: TransitionSampler,
    ) -> RewardCollection:
        total_mean_rewards = RewardCollection()
        indices = self.compute_batch_indices(states, transition_sampler)
        for start, end in zip(indices, indices[1:]):
            cur_states = states[start:end]
            batch_size = len(cur_states)

            # The actions at this first timestep are ignored b/c we're interested in the subsequent transition.
            _, cur_next_states, weights = transition_sampler.sample(cur_states)
            cur_next_states = cur_next_states.flatten(0, 1)

            # Compute the mean reward from those sampled next states.
            cur_next_states_mean_rewards = self.compute_mean_rewards(models, cur_next_states, transition_sampler)

            # Average over transitions from the current states to get the conditional, total mean reward.
            cur_next_states_mean_rewards = cur_next_states_mean_rewards.reshape(
                batch_size,
                transition_sampler.num_transitions_per_state,
            )
            cur_next_states_mean_rewards *= weights
            mean_rewards = cur_next_states_mean_rewards.mean(dim=-1)
            total_mean_rewards.append(mean_rewards)

        return total_mean_rewards

    def _compute_conditional_per_state_out_of_distribution_total_mean_rewards(
            self,
            models: ModelCollection,
            states: torch.Tensor,
            next_states: torch.Tensor,
            transition_sampler: TransitionSampler,
    ) -> RewardCollection:
        """Computes the total reward mean associated with states in an out-of-distribution manner.

        It is OOD wrt the transition function because S' ~ T(S'|s,A), but S'' ~ T(S''|s', A). It's OOD because
        S'' should be sampled wrt S' as opposed to the actual observed next state, s'.

        The total mean reward is computed in a computationally expensive manner where each S' sample is paired
        with each S'' sample for a given state, yielding an O(N^2) algorithm where N is the number of transitions
        sampled per state. There's also a quick version below.
        """
        total_mean_rewards = RewardCollection()
        indices = self.compute_batch_indices(states, transition_sampler)
        # Alias for readability.
        num_trans_per_state = transition_sampler.num_transitions_per_state
        for start, end in zip(indices, indices[1:]):
            cur_states = states[start:end]
            cur_next_states = next_states[start:end]
            batch_size = len(cur_states)

            # The actions at this first timestep are ignored b/c we're interested in the subsequent transition.
            _, cur_sampled_next_states, weights_next = transition_sampler.sample(cur_states)
            # Tile these next_states and repeat_interleave the next_next_states so that each next_state is associated
            # with each of the possible next_next_states originating from the same batch index.
            cur_sampled_next_states = cur_sampled_next_states.tile((1, num_trans_per_state, 1))
            cur_sampled_next_states = cur_sampled_next_states.flatten(0, 1)
            weights_next = weights_next.tile((1, num_trans_per_state)).reshape(
                -1,
                num_trans_per_state,
                num_trans_per_state,
            )

            # Sample the next, next states from the original next states to ensure their distribution matches that of S'' exactly.
            actions, cur_sampled_next_next_states, weights_next_next = transition_sampler.sample(cur_next_states)
            # Here we repeat_interleave instead of tile.
            actions = actions.repeat_interleave(num_trans_per_state, dim=1)
            actions = actions.flatten(0, 1)
            cur_sampled_next_next_states = cur_sampled_next_next_states.repeat_interleave(num_trans_per_state, dim=1)
            cur_sampled_next_next_states = cur_sampled_next_next_states.flatten(0, 1)
            weights_next_next = weights_next_next.repeat_interleave(num_trans_per_state, dim=1).reshape(
                -1,
                num_trans_per_state,
                num_trans_per_state,
            )

            cur_rewards = models.rewards(cur_sampled_next_states, actions, cur_sampled_next_next_states, terminals=None)
            # Average over transitions from the current states to get the conditional, total mean reward.
            cur_rewards = cur_rewards.reshape(batch_size, num_trans_per_state, num_trans_per_state)
            cur_rewards *= weights_next * weights_next_next
            mean_rewards = cur_rewards.mean(dim=(1, 2))
            total_mean_rewards.append(mean_rewards)

        return total_mean_rewards

    def _compute_conditional_per_state_out_of_distribution_total_mean_rewards_quick(
            self,
            models: ModelCollection,
            states: torch.Tensor,
            next_states: torch.Tensor,
            transition_sampler: TransitionSampler,
    ) -> RewardCollection:
        """Computes a quick version of the total mean reward.

            This is an O(N) algorithm for computing the total mean reward as opposed to O(N^2).
            See `_compute_conditional_per_state_out_of_distribution_total_mean_rewards` docs for more detail.
            """
        total_mean_rewards = RewardCollection()
        indices = self.compute_batch_indices(states, transition_sampler)
        for start, end in zip(indices, indices[1:]):
            cur_states = states[start:end]
            cur_next_states = next_states[start:end]
            batch_size = len(cur_states)

            # The actions at this first timestep are ignored b/c we're interested in the subsequent transition.
            _, cur_sampled_next_states, weights_next = transition_sampler.sample(cur_states)
            cur_sampled_next_states = cur_sampled_next_states.flatten(0, 1)

            # Sample the next, next states from the original next states to ensure their distribution matches that of S'' exactly.
            actions, cur_sampled_next_next_states, weights_next_next = transition_sampler.sample(cur_next_states)
            actions = actions.flatten(0, 1)
            cur_sampled_next_next_states = cur_sampled_next_next_states.flatten(0, 1)

            cur_rewards = models.rewards(cur_sampled_next_states, actions, cur_sampled_next_next_states, terminals=None)
            # Average over transitions from the current states to get the conditional, total mean reward.
            cur_rewards = cur_rewards.reshape(batch_size, transition_sampler.num_transitions_per_state)
            cur_rewards *= weights_next * weights_next_next
            mean_rewards = cur_rewards.mean(dim=-1)
            total_mean_rewards.append(mean_rewards)

        return total_mean_rewards


