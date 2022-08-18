"""Tests `imitation.util.networks`."""

import functools
import math
from typing import Type

import pytest
import torch as th

from imitation.util import networks

assert_equal = functools.partial(th.testing.assert_close, rtol=0, atol=0)


NORMALIZATION_LAYERS = [
    networks.RunningNorm,
    networks.EMANorm,
]


class EMANormIncremental(networks.EMANorm):
    """EMA Norm with every new instance added incrementally in a for loop."""

    def __init__(
        self,
        num_features: int,
        decay: float = 0.99,
        eps: float = 1e-5,
    ):
        """Builds EMARunningNormIncremental."""
        super().__init__(
            num_features,
            decay=decay,
            eps=eps,
        )

    def update_stats(self, batch: th.Tensor) -> None:
        """Update `self.running_mean` and `self.running_var` incrementally.

        Reference Finch (2009), "Incremental calculation of weighted mean and variance".
            (https://fanf2.user.srcf.net/hermes/doc/antiforgery/stats.pdf)

        Args:
            batch: A batch of data to use to update the running mean and variance.
        """
        b_size = batch.shape[0]
        if len(batch.shape) == 1:
            batch = batch.reshape(b_size, 1)
        alpha = 1 - self.decay
        if self.count == 0:
            self.running_mean = batch.mean(dim=0)
            if b_size > 1:
                self.running_var = batch.var(dim=0, unbiased=False)
            else:
                self.running_var = th.zeros_like(self.running_mean, dtype=th.float)
        else:
            # E[x^2] of previous data
            self.running_var += self.running_mean**2

            # update running mean
            diff = batch.mean(0) - self.running_mean
            self.running_mean += alpha * diff

            # update running variance
            sqdiff = (batch**2).mean(0) - self.running_var
            self.running_var += alpha * sqdiff
            self.running_var -= self.running_mean**2

        self.count += b_size


@pytest.mark.parametrize("normalization_layer", NORMALIZATION_LAYERS)
def test_running_norm_identity_eval(normalization_layer: Type[networks.BaseNorm]):
    """Tests running norm starts and stays at identity function when in eval mode.

    Specifically, we test in evaluation mode (initialization should not change)

    Args:
        normalization_layer: the normalization layer to be tested.
    """
    running_norm = normalization_layer(1, eps=0.0)
    x = th.Tensor([-1.0, 0.0, 7.32, 42.0])
    running_norm.eval()  # stats should not change in eval mode
    for _ in range(10):
        assert_equal(running_norm.forward(x), x)


def test_running_norm_identity_train():
    """Test that the running norm will not change already normalized data."""
    running_norm = networks.RunningNorm(1, eps=0.0)
    running_norm.train()  # stats will change in eval mode
    normalized = th.Tensor([-1, -1, -1, -1, 1, 1, 1, 1])  # mean 0, variance 1
    for _ in range(10):
        th.testing.assert_allclose(
            running_norm.forward(normalized),
            normalized,
            rtol=0.05,
            atol=0.05,
        )


@pytest.mark.parametrize("normalization_layer", NORMALIZATION_LAYERS)
def test_running_norm_eval_fixed(
    normalization_layer: Type[networks.BaseNorm],
    batch_size: int = 8,
    num_batches: int = 10,
    num_features: int = 4,
) -> None:
    """Tests that stats do not change when in eval mode and do when in training."""
    running_norm = normalization_layer(num_features)

    def do_forward(shift: float = 0.0, scale: float = 1.0):
        for i in range(num_batches):
            data = th.rand(batch_size, num_features) * scale + shift
            running_norm.forward(data)

    with th.random.fork_rng():
        th.random.manual_seed(42)

        do_forward()
        current_mean = th.clone(running_norm.running_mean)
        current_var = th.clone(running_norm.running_var)

        running_norm.eval()
        do_forward()
        assert_equal(running_norm.running_mean, current_mean)
        assert_equal(running_norm.running_var, current_var)

        running_norm.train()
        do_forward(1.0, 2.0)
        assert th.all((running_norm.running_mean - current_mean).abs() > 0.01)
        assert th.all((running_norm.running_var - current_var).abs() > 0.01)


@pytest.mark.parametrize("batch_size", [1, 8])
def test_running_norm_matches_dist(batch_size: int) -> None:
    """Test running norm converges to empirical distribution."""
    mean = th.Tensor([-1.3, 0.0, 42])
    var = th.Tensor([0.1, 1.0, 42])
    sd = th.sqrt(var)

    num_dims = len(mean)
    running_norm = networks.RunningNorm(num_dims)
    running_norm.train()

    num_samples = 256
    with th.random.fork_rng():
        th.random.manual_seed(42)
        data = th.randn(num_samples, num_dims) * sd + mean
        for start in range(0, num_samples, batch_size):
            batch = data[start : start + batch_size]
            running_norm.forward(batch)

    empirical_mean = th.mean(data, dim=0)
    empirical_var = th.var(data, dim=0, unbiased=False)

    normalized = th.Tensor([[-1.0], [0.0], [1.0], [42.0]])
    normalized = th.tile(normalized, (1, 3))
    scaled = normalized * th.sqrt(empirical_var + running_norm.eps) + empirical_mean
    running_norm.eval()
    for i in range(5):
        th.testing.assert_close(running_norm.forward(scaled), normalized)

    # Stats should match empirical mean (and be unchanged by eval)
    th.testing.assert_close(running_norm.running_mean, empirical_mean)
    th.testing.assert_close(running_norm.running_var, empirical_var)
    assert running_norm.count == num_samples


@pytest.mark.parametrize("batch_size", [1, 8])
@pytest.mark.parametrize("normalization_layer", NORMALIZATION_LAYERS)
def test_parameters_converge(
    batch_size: int,
    normalization_layer: Type[networks.BaseNorm],
) -> None:
    """Test running norm parameters approximately converge to true values."""
    mean = th.Tensor([3, 0])
    var = th.Tensor([6, 1])
    sd = th.sqrt(var)

    num_dims = len(mean)
    running_norm = normalization_layer(num_dims)
    running_norm.train()

    num_samples = 2000
    with th.random.fork_rng():
        th.random.manual_seed(42)
        data = th.randn(num_samples, num_dims) * sd + mean
        for start in range(0, num_samples, batch_size):
            batch = data[start : start + batch_size]
            running_norm.forward(batch)

    running_norm.eval()
    th.testing.assert_close(running_norm.running_mean, mean, rtol=0.05, atol=0.1)
    th.testing.assert_close(running_norm.running_var, var, rtol=0.1, atol=0.1)

    assert running_norm.count == num_samples


@pytest.mark.parametrize(
    "init_kwargs",
    [{}, {"dropout_prob": 0.5}]
    + [{"normalize_input_layer": layer} for layer in NORMALIZATION_LAYERS],
)
def test_build_mlp_norm_training(init_kwargs) -> None:
    """Tests MLP building function `networks.build_mlp()`.

    Specifically, we initialize an MLP and train it on a toy task. We also test the
    init options of input layer normalization.

    Args:
        init_kwargs: dict of kwargs to pass to `networks.build_mlp()`.
    """
    # Create Tensors to hold input and outputs.
    x = th.linspace(-math.pi, math.pi, 200).reshape(-1, 1)
    y = th.sin(x)
    # Construct our model by instantiating the class defined above
    model = networks.build_mlp(in_size=1, hid_sizes=[16, 16], out_size=1, **init_kwargs)

    # Construct a loss function and an Optimizer.
    criterion = th.nn.MSELoss(reduction="sum")
    optimizer = th.optim.SGD(model.parameters(), lr=1e-6)
    for t in range(200):
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)

        # Zero gradients, perform a backward pass, and update the weights.
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


def test_input_validation_on_ema_norm():
    with pytest.raises(ValueError):
        networks.EMANorm(128, decay=1.1)

    with pytest.raises(ValueError):
        networks.EMANorm(128, decay=-0.1)

    networks.EMANorm(128, decay=0.05)


@pytest.mark.parametrize("decay", [0.5, 0.99])
@pytest.mark.parametrize("input_shape", [(64,), (1, 256), (64, 256)])
def test_ema_norm_batch_correctness(decay, input_shape):
    norm_for_incremental = EMANormIncremental(
        num_features=256,
        decay=decay,
    )
    norm_for_batch = networks.EMANorm(256, decay)
    random_tensor = th.randn(input_shape)
    for i in range(100):
        norm_for_incremental.train(), norm_for_batch.train()
        # moving distribution
        tensor_input = random_tensor.clone() + i
        norm_for_incremental(tensor_input)
        tensor_input = random_tensor.clone() + i
        norm_for_batch(tensor_input)

        norm_for_incremental.eval(), norm_for_batch.eval()
        th.testing.assert_close(
            norm_for_incremental.running_mean,
            norm_for_batch.running_mean,
            rtol=0.05,
            atol=0.1,
        )
        th.testing.assert_close(
            norm_for_incremental.running_var,
            norm_for_batch.running_var,
            rtol=0.05,
            atol=0.1,
        )
