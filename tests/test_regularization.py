"""Tests for `imitation.regularization.*`."""
import itertools
import tempfile

import numpy as np
import pytest
import torch as th

from imitation.regularization import regularizers, updaters
from imitation.util import logger as imit_logger


@pytest.fixture(
    scope="module",
    params=[
        (0.5, (0.9, 1)),  # unlikely to fall inside the interval
        (0.5, (0.01, 10)),  # likely to fall inside the interval
    ],
)
def interval_param_scaler(request):
    return updaters.IntervalParamScaler(*request.param)


@pytest.mark.parametrize(
    "lambda_",
    [
        10.0,
        0.001,
    ],
)
@pytest.mark.parametrize(
    "train_loss",
    [
        th.tensor(100.0),
        th.tensor(10.0),
        th.tensor(0.1),
        th.tensor(0.0),
        100.0,
        10.0,
        0.1,
        0.0,
    ],
)
def test_interval_param_scaler(lambda_, train_loss, interval_param_scaler):
    scaler = interval_param_scaler
    tolerable_interval = scaler.tolerable_interval
    scaling_factor = scaler.scaling_factor
    eps = np.finfo(float).eps
    if train_loss > eps:
        # The loss is a non-zero scalar, so we can construct a validation loss for
        # three different cases:

        # case that the ratio between the validation loss and the training loss is
        # above the tolerable interval
        val_loss = train_loss * tolerable_interval[1] * 2
        assert scaler(lambda_, train_loss, val_loss) == lambda_ * (1 + scaling_factor)

        # case that the ratio between the validation loss and the training loss is
        # below the tolerable interval
        val_loss = train_loss * tolerable_interval[0] / 2
        assert scaler(lambda_, train_loss, val_loss) == lambda_ * (1 - scaling_factor)

        # case that the ratio between the validation loss and the training loss is
        # within the tolerable interval
        val_loss = train_loss * (tolerable_interval[0] + tolerable_interval[1]) / 2
        assert scaler(lambda_, train_loss, val_loss) == lambda_
    else:
        # we have a zero loss. We try two cases. When the validation loss is zero,
        # the ratio is undefined, so we should return the current lambda. When the
        # validation loss is nonzero, the ratio is infinite, so we should see the lambda
        # increase by the scaling factor.
        # We try it for both a tensor and a float value.
        val_loss = th.tensor(0.0)
        assert scaler(lambda_, train_loss, val_loss) == lambda_
        val_loss = 0.0
        assert scaler(lambda_, train_loss, val_loss) == lambda_
        val_loss = th.tensor(1.0)
        assert scaler(lambda_, train_loss, val_loss) == lambda_ * (1 + scaling_factor)
        val_loss = 1.0
        assert scaler(lambda_, train_loss, val_loss) == lambda_ * (1 + scaling_factor)


def test_interval_param_scaler_raises(interval_param_scaler):
    scaler = interval_param_scaler
    with pytest.raises(ValueError, match="val_loss must be a scalar"):
        scaler(1.0, 1.0, th.Tensor([3.0, 4.0]))
    with pytest.raises(ValueError, match="train_loss must be a scalar"):
        scaler(1.0, th.Tensor([1.0, 2.0]), 1.0)
    with pytest.raises(ValueError, match="train_loss must be a scalar"):
        scaler(1.0, "random value", th.tensor(1.0))
    with pytest.raises(ValueError, match="val_loss must be a scalar"):
        scaler(1.0, 1.0, "random value")
    with pytest.raises(ValueError, match="lambda_ must be a float"):
        scaler(th.tensor(1.0), 1.0, 1.0)
    with pytest.raises(ValueError, match="lambda_ must not be zero.*"):
        scaler(0.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="lambda_ must be non-negative.*"):
        scaler(-1.0, 1.0, 1.0)
    with pytest.raises(ValueError, match="losses must be non-negative.*"):
        scaler(1.0, -1.0, 1.0)
    with pytest.raises(ValueError, match="losses must be non-negative.*"):
        scaler(1.0, 1.0, -1.0)


def test_interval_param_scaler_init_raises():
    # this validates the value of scaling_factor.
    interval_err_msg = r"scaling_factor must be in \(0, 1\) within machine precision."

    with pytest.raises(ValueError, match=interval_err_msg):
        # cannot be negative as this is counter-intuitive to
        # the direction of scaling (just use the reciprocal).
        updaters.IntervalParamScaler(-1, (0.1, 0.9))

    with pytest.raises(ValueError, match=interval_err_msg):
        # cannot be larger than one as this would make lambda
        # negative when scaling down.
        updaters.IntervalParamScaler(1.1, (0.1, 0.9))

    with pytest.raises(ValueError, match=interval_err_msg):
        # cannot be exactly zero, as this never changes the value
        # of lambda when scaling up.
        updaters.IntervalParamScaler(0.0, (0.1, 0.9))

    with pytest.raises(ValueError, match=interval_err_msg):
        # cannot be exactly one, as when lambda is scaled down
        # this brings it to zero.
        updaters.IntervalParamScaler(1.0, (0.1, 0.9))

    # an interval obviously needs two elements only.
    with pytest.raises(
        ValueError,
        match="tolerable_interval must be a tuple of length 2",
    ):
        updaters.IntervalParamScaler(0.5, (0.1, 0.9, 0.5))  # type: ignore[arg-type]
    with pytest.raises(
        ValueError,
        match="tolerable_interval must be a tuple of length 2",
    ):
        updaters.IntervalParamScaler(0.5, (0.1,))  # type: ignore[arg-type]

    # the first element of the interval must be at least 0.
    with pytest.raises(
        ValueError,
        match="tolerable_interval must be a tuple whose first element "
        "is at least 0.*",
    ):
        updaters.IntervalParamScaler(0.5, (-0.1, 0.9))

    # the second element of the interval must be greater than the first.
    with pytest.raises(
        ValueError,
        match="tolerable_interval must be a tuple.*the second "
        "element is greater than the first",
    ):
        updaters.IntervalParamScaler(0.5, (0.1, 0.05))


@pytest.fixture(scope="module")
def hierarchical_logger():
    tmpdir = tempfile.mkdtemp()
    return imit_logger.configure(tmpdir, ["tensorboard", "stdout", "csv"])


@pytest.fixture(scope="module", params=[0.1, 1.0, 10.0])
def simple_optimizer(request):
    return th.optim.Adam([th.tensor(request.param, requires_grad=True)], lr=0.1)


@pytest.fixture(scope="module", params=[0.1, 1.0, 10.0])
def initial_lambda(request):
    return request.param


class SimpleRegularizer(regularizers.Regularizer[None]):
    """A simple regularizer that does nothing."""

    def regularize_and_backward(self, loss: th.Tensor) -> None:
        pass  # pragma: no cover


def test_regularizer_init_no_crash(
    initial_lambda,
    hierarchical_logger,
    simple_optimizer,
    interval_param_scaler,
):
    SimpleRegularizer(
        initial_lambda=initial_lambda,
        optimizer=simple_optimizer,
        logger=hierarchical_logger,
        lambda_updater=interval_param_scaler,
        val_split=0.2,
    )

    SimpleRegularizer(
        initial_lambda=initial_lambda,
        optimizer=simple_optimizer,
        logger=hierarchical_logger,
        lambda_updater=None,
        val_split=None,
    )

    SimpleRegularizer.create(
        initial_lambda=initial_lambda,
        lambda_updater=interval_param_scaler,
        val_split=0.2,
    )(
        optimizer=simple_optimizer,
        logger=hierarchical_logger,
    )


@pytest.mark.parametrize(
    "val_split",
    [
        0.0,
        1.0,
        -10,
        10,
        "random value",
        10**-100,
    ],
)
def test_regularizer_init_raises_on_val_split(
    initial_lambda,
    hierarchical_logger,
    simple_optimizer,
    interval_param_scaler,
    val_split,
):
    val_split_err_msg = "val_split.*must be a float.*between.*"
    with pytest.raises(ValueError, match=val_split_err_msg):
        return SimpleRegularizer(
            initial_lambda=initial_lambda,
            optimizer=simple_optimizer,
            logger=hierarchical_logger,
            lambda_updater=interval_param_scaler,
            val_split=val_split,
        )


def test_regularizer_init_raises(
    initial_lambda,
    hierarchical_logger,
    simple_optimizer,
    interval_param_scaler,
):
    with pytest.raises(
        ValueError,
        match=".*do not pass.*parameter updater.*regularization strength.*non-zero",
    ):
        SimpleRegularizer(
            initial_lambda=0.0,
            optimizer=simple_optimizer,
            logger=hierarchical_logger,
            lambda_updater=None,
            val_split=0.2,
        )
    with pytest.raises(
        ValueError,
        match=".*pass.*parameter updater.*must.*specify.*validation split.*",
    ):
        SimpleRegularizer(
            initial_lambda=initial_lambda,
            optimizer=simple_optimizer,
            logger=hierarchical_logger,
            lambda_updater=interval_param_scaler,
            val_split=None,
        )
    with pytest.raises(
        ValueError,
        match=".*pass.*validation split.*must.*pass.*parameter updater.*",
    ):
        SimpleRegularizer(
            initial_lambda=initial_lambda,
            optimizer=simple_optimizer,
            logger=hierarchical_logger,
            lambda_updater=None,
            val_split=0.2,
        )


@pytest.mark.parametrize(
    "train_loss",
    [
        th.tensor(10.0),
        th.tensor(1.0),
        th.tensor(0.1),
        th.tensor(0.01),
    ],
)
def test_regularizer_update_params(
    initial_lambda,
    hierarchical_logger,
    simple_optimizer,
    interval_param_scaler,
    train_loss,
):
    regularizer = SimpleRegularizer(
        initial_lambda=initial_lambda,
        logger=hierarchical_logger,
        lambda_updater=interval_param_scaler,
        optimizer=simple_optimizer,
        val_split=0.1,
    )
    val_to_train_loss_ratio = interval_param_scaler.tolerable_interval[1] * 2
    val_loss = train_loss * val_to_train_loss_ratio
    assert regularizer.lambda_ == initial_lambda
    assert (
        hierarchical_logger.default_logger.name_to_value["regularization_lambda"]
        == initial_lambda
    )
    regularizer.update_params(train_loss, val_loss)
    expected_lambda_value = interval_param_scaler(initial_lambda, train_loss, val_loss)
    assert regularizer.lambda_ == expected_lambda_value
    assert expected_lambda_value != initial_lambda
    assert (
        hierarchical_logger.default_logger.name_to_value["regularization_lambda"]
        == expected_lambda_value
    )


class SimpleLossRegularizer(regularizers.LossRegularizer):
    """A simple loss regularizer.

    It multiplies the total loss by lambda_+1.
    """

    def _loss_penalty(self, loss: regularizers.Scalar) -> regularizers.Scalar:
        return loss * self.lambda_  # this multiplies the total loss by lambda_+1.


@pytest.mark.parametrize(
    "train_loss_base",
    [
        th.tensor(10.0),
        th.tensor(1.0),
        th.tensor(0.1),
        th.tensor(0.01),
    ],
)
def test_loss_regularizer(
    hierarchical_logger,
    simple_optimizer,
    initial_lambda,
    train_loss_base,
):
    regularizer = SimpleLossRegularizer(
        initial_lambda=initial_lambda,
        logger=hierarchical_logger,
        lambda_updater=None,
        optimizer=simple_optimizer,
    )
    loss_param = simple_optimizer.param_groups[0]["params"][0]
    train_loss = train_loss_base * loss_param
    regularizer.optimizer.zero_grad()
    regularized_loss = regularizer.regularize_and_backward(train_loss)
    assert th.allclose(regularized_loss.data, train_loss * (initial_lambda + 1))
    assert (
        hierarchical_logger.default_logger.name_to_value["regularized_loss"]
        == regularized_loss
    )
    assert th.allclose(loss_param.grad, train_loss_base * (initial_lambda + 1))


class SimpleWeightRegularizer(regularizers.WeightRegularizer):
    """A simple weight regularizer.

    It multiplies the total weight by lambda_+1.
    """

    def _weight_penalty(self, weight, group):
        # this multiplies the total weight by lambda_+1.
        # However, the grad is only calculated with respect to the
        # previous value of the weight.
        # This difference is only noticeable if the grad of the loss
        # has a functional dependence on the weight (i.e. not linear).
        return weight * self.lambda_


@pytest.mark.parametrize(
    "train_loss_base",
    [
        th.tensor(10.0),
        th.tensor(1.0),
        th.tensor(0.1),
        th.tensor(0.01),
    ],
)
def test_weight_regularizer(
    hierarchical_logger,
    simple_optimizer,
    initial_lambda,
    train_loss_base,
):
    regularizer = SimpleWeightRegularizer(
        initial_lambda=initial_lambda,
        logger=hierarchical_logger,
        lambda_updater=None,
        optimizer=simple_optimizer,
    )
    weight = simple_optimizer.param_groups[0]["params"][0]
    initial_weight_value = weight.data.clone()
    regularizer.optimizer.zero_grad()
    train_loss = train_loss_base * th.pow(weight, 2) / 2
    regularizer.regularize_and_backward(train_loss)
    assert th.allclose(weight.data, initial_weight_value * (initial_lambda + 1))
    assert th.allclose(weight.grad, train_loss_base * initial_weight_value)


@pytest.mark.parametrize("p", [0.5, 1.5, -1, 0, "random value"])
def test_lp_regularizer_p_value_raises(hierarchical_logger, simple_optimizer, p):
    with pytest.raises(ValueError, match="p must be a positive integer"):
        regularizers.LpRegularizer(
            initial_lambda=1.0,
            logger=hierarchical_logger,
            lambda_updater=None,
            optimizer=simple_optimizer,
            p=p,
        )


MULTI_PARAM_OPTIMIZER_INIT_VALS = [-1.0, 0.0, 1.0]
MULTI_PARAM_OPTIMIZER_ARGS = itertools.product(
    MULTI_PARAM_OPTIMIZER_INIT_VALS,
    MULTI_PARAM_OPTIMIZER_INIT_VALS,
)


@pytest.fixture(scope="module", params=MULTI_PARAM_OPTIMIZER_ARGS)
def multi_param_optimizer(request):
    return th.optim.Adam(
        [th.tensor(p, requires_grad=True) for p in request.param],
        lr=0.1,
    )


MULTI_PARAM_AND_LR_OPTIMIZER_ARGS = itertools.product(
    MULTI_PARAM_OPTIMIZER_INIT_VALS,
    MULTI_PARAM_OPTIMIZER_INIT_VALS,
    [0.001, 0.01, 0.1],
)


@pytest.fixture(scope="module", params=MULTI_PARAM_AND_LR_OPTIMIZER_ARGS)
def multi_param_and_lr_optimizer(request):
    return th.optim.Adam(
        [th.tensor(p, requires_grad=True) for p in request.param[:-1]],
        lr=request.param[-1],
    )


@pytest.mark.parametrize(
    "train_loss",
    [
        th.tensor(10.0),
        th.tensor(1.0),
        th.tensor(0.1),
    ],
)
@pytest.mark.parametrize("p", [1, 2, 3])
def test_lp_regularizer(
    hierarchical_logger,
    multi_param_optimizer,
    initial_lambda,
    train_loss,
    p,
):
    regularizer = regularizers.LpRegularizer(
        initial_lambda=initial_lambda,
        logger=hierarchical_logger,
        lambda_updater=None,
        optimizer=multi_param_optimizer,
        p=p,
    )
    params = multi_param_optimizer.param_groups[0]["params"]
    regularizer.optimizer.zero_grad()
    regularized_loss = regularizer.regularize_and_backward(train_loss)
    loss_penalty = sum(
        [th.linalg.vector_norm(param.data, ord=p).pow(p) for param in params],
    )
    assert th.allclose(
        regularized_loss.data,
        train_loss + initial_lambda * loss_penalty,
    )
    assert (
        regularized_loss
        == hierarchical_logger.default_logger.name_to_value["regularized_loss"]
    )
    for param in params:
        assert th.allclose(
            param.grad,
            p * initial_lambda * th.abs(param).pow(p - 1) * th.sign(param),
        )


@pytest.mark.parametrize(
    "train_loss_base",
    [
        th.tensor(1.0),
        th.tensor(0.1),
        th.tensor(0.01),
    ],
)
def test_weight_decay_regularizer(
    multi_param_and_lr_optimizer,
    hierarchical_logger,
    initial_lambda,
    train_loss_base,
):
    regularizer = regularizers.WeightDecayRegularizer(
        initial_lambda=initial_lambda,
        logger=hierarchical_logger,
        lambda_updater=None,
        optimizer=multi_param_and_lr_optimizer,
    )
    weights = regularizer.optimizer.param_groups[0]["params"]
    lr = regularizer.optimizer.param_groups[0]["lr"]
    initial_weight_values = [weight.data.clone() for weight in weights]
    regularizer.optimizer.zero_grad()
    train_loss = train_loss_base * sum(th.pow(weight, 2) / 2 for weight in weights)
    regularizer.regularize_and_backward(train_loss)
    for weight, initial_weight_value in zip(weights, initial_weight_values):
        assert th.allclose(
            weight.data,
            initial_weight_value * (1 - lr * initial_lambda),
        )
        assert th.allclose(weight.grad, train_loss_base * initial_weight_value)
