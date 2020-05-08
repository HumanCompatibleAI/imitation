"""Helper methods to build and run neural networks."""

import collections
import contextlib
from typing import Callable, Dict, Iterable, Optional, Sequence, Tuple

import gym
import tensorflow as tf
from stable_baselines.common.input import observation_input

LayersDict = Dict[str, tf.layers.Layer]


def build_mlp(
    hid_sizes: Iterable[int],
    name: Optional[str] = None,
    activation: Optional[Callable] = tf.nn.relu,
    initializer: Optional[Callable] = None,
) -> LayersDict:
    """Constructs an MLP, returning an ordered dict of layers."""
    layers = collections.OrderedDict()

    # Hidden layers
    for i, size in enumerate(hid_sizes):
        key = f"{name}_dense{i}"
        layer = tf.layers.Dense(
            size, activation=activation, kernel_initializer=initializer, name=key
        )  # type: tf.layers.Layer
        layers[key] = layer

    # Final layer
    layer = tf.layers.Dense(
        1, kernel_initializer=initializer, name=f"{name}_dense_final"
    )  # type: tf.layers.Layer
    layers[f"{name}_dense_final"] = layer

    return layers


def sequential(inputs: tf.Tensor, layers: LayersDict) -> tf.Tensor:
    """Applies a sequence of layers to an input."""
    output = inputs
    for layer in layers.values():
        output = layer(output)
    output = tf.squeeze(output, axis=1)
    return output


def build_inputs(
    observation_space: gym.Space, action_space: gym.Space, scale: bool = False
) -> Tuple[tf.Tensor, ...]:
    """Builds placeholders and processed input Tensors.

    Observation `obs_*` and `next_obs_*` placeholders and processed input
    tensors have shape `(None,) + obs_space.shape`.
    The action `act_*` placeholder and processed input tensors have shape
    `(None,) + act_space.shape`.

    Args:
        observation_space: The observation space.
        action_space: The action space.
        scale: Only relevant for environments with Box spaces. If True, then
            processed input Tensors are automatically scaled to the interval [0, 1].

    Returns:
        (phs, inps) where phs is a tuple of:
            obs_ph: Placeholder for old observations.
            act_ph: Placeholder for actions.
            next_obs_ph: Placeholder for new observations.
            done_ph: Placeholder for boolean episode termination.
        and inps is a tuple of:
            obs_inp: Network-ready float32 Tensor with processed old observations.
            act_inp: Network-ready float32 Tensor with processed actions.
            next_obs_inp: Network-ready float32 Tensor with processed new observations.
            dones_inp: Network-ready float32 tensor, with booleans 0-1 coded.
    """
    obs_ph, obs_inp = observation_input(observation_space, name="obs", scale=scale)
    act_ph, act_inp = observation_input(action_space, name="act", scale=scale)
    next_obs_ph, next_obs_inp = observation_input(
        observation_space, name="next_obs", scale=scale
    )
    done_ph = tf.placeholder(name="dones", shape=(None,), dtype=tf.bool)
    done_inp = tf.cast(done_ph, dtype=tf.float32)
    phs = (obs_ph, act_ph, next_obs_ph, done_ph)
    inps = (obs_inp, act_inp, next_obs_inp, done_inp)
    return phs, inps


@contextlib.contextmanager
def make_session(close_on_exit: bool = True, **kwargs):
    """Context manager for a TensorFlow session.

    The session is associated with a newly created graph. Both session and
    graph are set as default. The session will be closed when exiting this
    context manager.

    Args:
      close_on_exit: If True, closes the session upon leaving the context manager.
      kwargs: passed through to `tf.Session`.

    Yields:
      (graph, session) where graph is a `tf.Graph` and `session` a `tf.Session`.
    """
    graph = tf.Graph()
    with graph.as_default():
        session = tf.Session(graph=graph, **kwargs)
        try:
            with session.as_default():
                yield graph, session
        finally:
            if close_on_exit:
                session.close()


def build_and_apply_mlp(
    inputs: Sequence[tf.Tensor],
    hid_sizes: Optional[Iterable[int]] = None,
    **kwargs: dict,
) -> Tuple[tf.Tensor, LayersDict]:
    """Builds an MLP depending on specified inputs.

    All specified inputs will be flattened and then concatenated. They are then
    applied using `sequential` to an MLP built using `build_mlp`.

    Arguments:
        hid_sizes: Number of units at each hidden layer.
            Default is (32, 32), i.e. two hidden layers with 32 units.
            () represents linear.
      inputs: Sequence of tensor inputs to flatten and concatenate.
      **kwargs: Passed through to `util.build_mlp`.

    Returns:
      (output, mlp) where output is the predicted reward and mlp is a LayersDict.

    Raises:
      ValueError: If inputs is a length-0 tensor.
    """
    if len(inputs) == 0:
        raise ValueError("Must specify at least one input")

    if hid_sizes is None:
        hid_sizes = (32, 32)

    with tf.variable_scope("theta"):
        inputs = [tf.layers.flatten(x) for x in inputs]
        inputs = tf.concat(inputs, axis=1)
        mlp = build_mlp(hid_sizes=hid_sizes, name="reward", **kwargs)
        output = sequential(inputs, mlp)

        return output, mlp
