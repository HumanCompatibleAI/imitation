"""Tests for `imitation.regularization.*`."""

import numpy as np
import pytest
import torch as th

from imitation.regularization import updaters

CONSTANT_PARAM_SCALER_TEST_ARGS = [
    (10., 0, 0),
    (0.1, 0, 0),
    (10., 1., 1.),
]


@pytest.mark.parametrize("lambda_,train_loss,val_loss", CONSTANT_PARAM_SCALER_TEST_ARGS)
def test_constant_param_scaler(lambda_, train_loss, val_loss):
    scaler = updaters.ConstantParamScaler()
    assert scaler(lambda_, train_loss, val_loss) == lambda_


@pytest.fixture(params=[
    (0.5, (0.5, 1)),
    (0.75, (0.01, 10)),
])
def interval_param_scaler(request):
    return updaters.IntervalParamScaler(*request.param)


@pytest.mark.parametrize("lambda_", [
    10.,
    0.001,
])
@pytest.mark.parametrize("train_loss", [
    th.tensor(10.),
    th.tensor(0.1),
    th.tensor(0.0),
    10.,
    0.1,
    0.0,
])
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
        val_loss = th.tensor(0.)
        assert scaler(lambda_, train_loss, val_loss) == lambda_
        val_loss = 0.
        assert scaler(lambda_, train_loss, val_loss) == lambda_
        val_loss = th.tensor(1.)
        assert scaler(lambda_, train_loss, val_loss) == lambda_ * (1 + scaling_factor)
        val_loss = 1.
        assert scaler(lambda_, train_loss, val_loss) == lambda_ * (1 + scaling_factor)


def test_interval_param_scaler_raises(interval_param_scaler):
    scaler = interval_param_scaler
    with pytest.raises(ValueError, match="val_loss and train_loss must be scalars"):
        scaler(0., th.Tensor([1., 2.]), th.Tensor([3., 4.]))
    with pytest.raises(ValueError, match="val_loss and train_loss must be scalars"):
        scaler(0., "random value", "random value")  # type: ignore
    with pytest.raises(ValueError, match="lambda_ must be a float"):
        scaler(th.tensor(1.0), 1., 1.)  # type: ignore
    with pytest.raises(ValueError, match="lambda_ must not be zero.*"):
        scaler(0., 1., 1.)


def test_interval_param_scaler_init_raises():
    # this validates the value of scaling_factor.
    with pytest.raises(ValueError, match=r"scaling_factor must be in \(0, 1\) within machine precision."):
        # cannot be negative as this is counter-intuitive to
        # the direction of scaling (just use the reciprocal).
        updaters.IntervalParamScaler(-1, (0.1, 0.9))
    with pytest.raises(ValueError, match=r"scaling_factor must be in \(0, 1\) within machine precision."):
        # cannot be larger than one as this would make lambda negative when scaling down.
        updaters.IntervalParamScaler(1.1, (0.1, 0.9))
    with pytest.raises(ValueError, match=r"scaling_factor must be in \(0, 1\) within machine precision."):
        # cannot be exactly zero, as this never changes the value of lambda when scaling up.
        updaters.IntervalParamScaler(0., (0.1, 0.9))
    with pytest.raises(ValueError, match=r"scaling_factor must be in \(0, 1\) within machine precision."):
        # cannot be exactly one, as when lambda is scaled down this brings it to zero.
        updaters.IntervalParamScaler(1., (0.1, 0.9))

    # an interval obviously needs two elements only.
    with pytest.raises(ValueError, match="tolerable_interval must be a tuple of length 2"):
        updaters.IntervalParamScaler(0.5, (0.1, 0.9, 0.5))  # type: ignore
    with pytest.raises(ValueError, match="tolerable_interval must be a tuple of length 2"):
        updaters.IntervalParamScaler(0.5, (0.1,))  # type: ignore

    # the first element of the interval must be at least 0.
    with pytest.raises(ValueError, match="tolerable_interval must be a tuple whose first element is at least 0.*"):
        updaters.IntervalParamScaler(0.5, (-0.1, 0.9))

    # the second element of the interval must be greater than the first.
    with pytest.raises(ValueError,
                       match="tolerable_interval must be a tuple.*the second element is greater than the first"):
        updaters.IntervalParamScaler(0.5, (0.1, 0.05))
