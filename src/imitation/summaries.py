import os.path

import tensorflow as tf


def make_summary_writer(log_dir, graph=None):
  tf.logging.info("building summary directory at " + log_dir)
  os.makedirs(log_dir, exist_ok=True)
  summary_writer = tf.summary.FileWriter(log_dir, graph=graph)
  return summary_writer
