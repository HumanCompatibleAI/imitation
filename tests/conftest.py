"""Configuration settings and fixtures for tests."""

import pytest
import tensorflow as tf


@pytest.fixture
def graph():
  graph = tf.Graph()
  with graph.as_default():
    yield graph


@pytest.fixture
def session(graph):
  with tf.Session(graph=graph) as sess:
    with sess.as_default():
      yield sess
