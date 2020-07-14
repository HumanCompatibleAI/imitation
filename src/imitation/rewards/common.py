"""Utilities and definitions shared by reward-related code."""

import collections
import functools
from typing import Callable, Dict, List, Tuple

import numpy as np
import torch as th
from stable_baselines3.common import vec_env

RewardFn = Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], np.ndarray]


def _reward_fn_normalize_inputs(
    obs: np.ndarray,
    acts: np.ndarray,
    next_obs: np.ndarray,
    dones: np.ndarray,
    *,
    reward_fn: RewardFn,
    vec_normalize: vec_env.VecNormalize,
    norm_reward: bool = True,
) -> np.ndarray:
    """Combine with `functools.partial` to create an input-normalizing RewardFn.

    Args:
        reward_fn: The reward function that normalized inputs are evaluated on.
        vec_normalize: Instance of VecNormalize used to normalize inputs and
            rewards.
        norm_reward: If True, then also normalize reward before returning.

    Returns:
        The possibly normalized reward.
    """
    norm_obs = vec_normalize.normalize_obs(obs)
    norm_next_obs = vec_normalize.normalize_obs(next_obs)
    rew = reward_fn(norm_obs, acts, norm_next_obs, dones)
    if norm_reward:
        rew = vec_normalize.normalize_reward(rew)
    return rew


def build_norm_reward_fn(*, reward_fn, vec_normalize, **kwargs) -> RewardFn:
    """Reward function that automatically normalizes inputs.

    See _reward_fn_normalize_inputs for argument documentation.
    """
    return functools.partial(
        _reward_fn_normalize_inputs,
        reward_fn=reward_fn,
        vec_normalize=vec_normalize,
        **kwargs,
    )


def compute_train_stats(
    disc_logits_gen_is_high: th.Tensor,
    labels_gen_is_one: th.Tensor,
    disc_loss: th.Tensor,
) -> Dict[str, float]:
    """Train statistics for GAIL/AIRL discriminator, or other binary classifiers.

    Args:
        disc_logits_gen_is_high: discriminator logits produced by
            `DiscrimNet.logits_gen_is_high`.
        labels_gen_is_one: integer labels describing whether logit was for an
            expert (0) or generator (1) sample.
        disc_loss: final discriminator loss.

    Returns:
        stats: dictionary mapping statistic names for float values."""
    with th.no_grad():
        bin_is_generated_pred = disc_logits_gen_is_high > 0
        bin_is_generated_true = labels_gen_is_one > 0
        bin_is_expert_true = th.logical_not(bin_is_generated_true)
        int_is_generated_pred = bin_is_generated_pred.long()
        int_is_generated_true = bin_is_generated_true.long()
        n_generated = th.sum(int_is_generated_true).item()
        n_labels = labels_gen_is_one.size(0)
        n_expert = n_labels - n_generated
        pct_expert = n_expert / float(n_labels)
        n_expert_pred = int(len(bin_is_generated_pred) - th.sum(int_is_generated_pred))
        pct_expert_pred = n_expert_pred / float(len(n_labels))
        correct_vec = th.eq(bin_is_generated_pred, bin_is_generated_true)
        acc = th.mean(correct_vec.float())

        _n_pred_expert = th.sum(th.logical_and(bin_is_expert_true, correct_vec))
        _n_expert_or_1 = max(1, n_expert)
        expert_acc = _n_pred_expert / _n_expert_or_1

        _n_pred_gen = th.sum(th.logical_and(bin_is_generated_true, correct_vec))
        _n_gen_or_1 = max(1, n_generated)
        generated_acc = _n_pred_gen / _n_gen_or_1

        label_dist = th.distributions.Bernoulli(disc_logits_gen_is_high)
        entropy = th.mean(label_dist.entropy())

    pairs = [
        ("disc_xent_loss", th.mean(disc_loss)),
        # accuracy, as well as accuracy on *just* expert examples and *just*
        # generated examples
        ("disc_acc", acc),
        ("disc_acc_expert", expert_acc),
        ("disc_acc_gen", generated_acc),
        # entropy of the predicted label distribution, averaged equally across
        # both classes (if this drops then disc is very good or has given up)
        ("disc_entropy", entropy),
        # true number of expert demos and predicted number of expert demos
        ("disc_proportion_expert_true", pct_expert),
        ("disc_proportion_expert_pred", pct_expert_pred),
        ("n_expert", n_expert),
        ("n_generated", n_generated),
    ]  # type: List[Tuple[str, th.Tensor]]
    # convert to float
    pairs = [
        (key, float(value.item()) if isinstance(value, th.Tensor) else value)
        for key, value in pairs
    ]
    stats = collections.OrderedDict(pairs)

    return stats
