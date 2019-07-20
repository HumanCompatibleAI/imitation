import os.path as osp

from sacred.observers import FileStorageObserver
from stable_baselines import logger as sb_logger
import tensorflow as tf

from imitation.scripts.config.data_collect import data_collect_ex
import imitation.util as util


@data_collect_ex.main
def main(_seed, env_name, log_dir, parallel, total_timesteps,
         num_vec=8, make_blank_policy_kwargs={}):
  tf.logging.set_verbosity(tf.logging.INFO)
  sb_logger.configure(folder=osp.join(log_dir, 'rl'),
                      format_strs=['tensorboard', 'stdout'])

  env = util.make_vec_env(env_name, num_vec, seed=_seed,
                          parallel=parallel, log_dir=log_dir)
  # TODO(adam): add support for wrapping env with VecNormalize
  # (This is non-trivial since we'd need to make sure it's also applied
  # when the policy is re-loaded to generate rollouts.)
  policy = util.make_blank_policy(env, verbose=1, **make_blank_policy_kwargs)

  callback = util.make_save_policy_callback(osp.join(log_dir, 'checkpoints'))
  policy.learn(total_timesteps, callback=callback)


if __name__ == "__main__":
    observer = FileStorageObserver.create(
        osp.join('output', 'sacred', 'data_collect'))
    data_collect_ex.observers.append(observer)
    data_collect_ex.run_commandline()
