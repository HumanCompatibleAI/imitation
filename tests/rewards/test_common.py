"""Unit tests for `imitation.rewards.common`."""

import numpy as np
import pytest
import torch as th

from imitation.rewards import common


@pytest.mark.parametrize("n_samples", [0, 1, 10, 40])
def test_compute_train_stats(n_samples):
    disc_logits_gen_is_high = th.from_numpy(np.random.standard_normal([n_samples]) * 10)
    labels_gen_is_one = th.from_numpy(np.random.randint(2, size=[n_samples]))
    disc_loss = th.tensor(np.random.random() * 10)
    stats = common.compute_train_stats(
        disc_logits_gen_is_high, labels_gen_is_one, disc_loss
    )
    for k, v in stats.items():
        assert isinstance(k, str)
        assert isinstance(v, float)
