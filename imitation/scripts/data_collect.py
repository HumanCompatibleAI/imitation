import os
import os.path as osp
from typing import Callable, Optional

import numpy as np
from sacred.observers import FileStorageObserver
import tensorflow as tf

from imitation.scripts.config.data_collect import data_collect_ex
import imitation.util as util


def make_PPO2(env_name, num_vec, **make_blank_policy_kwargs):
  env = util.make_vec_env(env_name, num_vec)
  # TODO(adam): add support for wrapping env with VecNormalize
  # (This is non-trivial since we'd need to make sure it's also applied
  # when the policy is re-loaded to generate rollouts.)
  policy = util.make_blank_policy(env, verbose=1, init_tensorboard=True,
                                  **make_blank_policy_kwargs)
  return policy


@data_collect_ex.main
def main(env_name: str,
         total_timesteps: int,
         *,
         num_vec: int = 8,
         make_blank_policy_kwargs: dict = {},

         rollout_save: bool = False,
         rollout_save_interval: Optional[int] = None,
         rollout_save_n_samples: Optional[int] = None,
         rollout_dir: Optional[str] = None,

         policy_save: bool = False,
         policy_save_interval: Optional[int] = None,
         policy_dir: Optional[str] = None,
         ):
  tf.logging.set_verbosity(tf.logging.INFO)

  policy = make_PPO2(env_name, num_vec, **make_blank_policy_kwargs)

  callback = _make_callback(
    rollout_save, rollout_save_interval, rollout_save_n_samples,
    rollout_dir, policy_save, policy_save_interval, policy_dir)
  policy.learn(total_timesteps, callback=callback)


def _make_callback(rollout_save: bool = False,
                   rollout_save_interval: Optional[int] = None,
                   rollout_save_n_samples: Optional[int] = None,
                   rollout_dir: Optional[str] = None,

                   policy_save: bool = False,
                   policy_save_interval: Optional[int] = None,
                   policy_dir: Optional[str] = None,
                   ) -> Callable:
  """Make a callback that saves policy weights and rollouts during training.

  At applicable training steps `step`,
    - Policies are saved to `{policies_dir}/{env_name}-{step}.pkl`.
    - Rollouts are saved to `{rollout_dir}/{env_name}-{step}.npz`.

  Args:
      rollout_save: Whether to save rollout files.
      rollout_save_interval: The number of training updates in between
          saving .npz rollout files.
      rollout_save_n_samples: The number of timesteps to generate
          and store in every .npz file.
      rollout_dir: The directory that rollouts should be saved to.

      policy_save: Whether to save policy weights.
      policy_save_interval: The number of training updates in between
          saving .pkl policy weight files.
      policy_dir: The directory that policies should be saved to.
  """
  step = 0
  os.makedirs(rollout_dir, exist_ok=True)
  os.makedirs(policy_dir, exist_ok=True)

  def callback(locals_: dict, _) -> bool:
    nonlocal step
    step += 1
    policy = locals_['self']
    env = policy.get_env()
    assert env is not None

    if rollout_save and step % rollout_save_interval == 0:
      filename = util.dump_prefix(policy.__class__, env, step) + ".npz"
      path = osp.join(rollout_dir, filename)
      obs_old, act, obs_new, rew = util.rollout.generate_transitions(
        policy, env, n_timesteps=rollout_save_n_samples)
      np.savez_compressed(path,
                          obs_old=obs_old, act=act, obs_new=obs_new, rew=rew)
      tf.logging.info("Dumped demonstrations to {}.".format(path))

    if policy_save and step % policy_save_interval == 0:
      filename = util.dump_prefix(policy.__class__, env, step) + ".pkl"
      path = osp.join(policy_dir, filename)
      policy.save(path)
      tf.logging.info("Saved policy pickle to {}.".format(path))
    return True

  return callback


if __name__ == "__main__":
    observer = FileStorageObserver.create(
        osp.join('output', 'sacred', 'data_collect'))
    data_collect_ex.observers.append(observer)
    data_collect_ex.run_commandline()
