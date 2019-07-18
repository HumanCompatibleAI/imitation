import os
import os.path as osp
from typing import Callable, Optional

import numpy as np
from sacred.observers import FileStorageObserver
import stable_baselines
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
         rollout_save_n_samples: int = 2000,
         rollout_dir: str = "data/rollouts",

         policy_save: bool = False,
         policy_save_interval: Optional[int] = None,
         policy_dir: str = "data/policies",
         ):
  """Train a policy from scratch, optionally saving the policy and rollouts.

  At applicable training steps `step`,
    - Policies are saved to
      `{policies_dir}/{env_name}-{policy_name}-{step}.pkl`.
    - Rollouts are saved to `{rollout_dir}/{env_name}-{policy_name}-{step}.npz`.

  Args:
      env_name: The gym.Env name. Loaded as VecEnv.
      total_timesteps: Number of training timesteps in `model.learn()`.
      num_vec: Number of environments in VecEnv.
      make_blank_policy_kwargs: Kwargs for `make_blank_policy`.
      rollout_save: Whether to save rollout files. If rollout_save is False,
          then all the other `rollout_*` arguments are ignored.
      rollout_save_interval: The number of training updates in between saves
          after the first save. If the argument is `None`, then only save the
          final update. Otherwise if the argument is an integer, then save
          rollouts every `rollout_save_interval` updates and after the final
          update.
      rollout_save_n_samples: The number of timesteps saved in every file.
      rollout_dir: The directory that rollouts are saved in.
      policy_save: Whether to save policy files. If policy_save is False,
          then all other `policy_*` arguments are ignored.
      policy_save_interval: The number of training updates between saves. Has
          the same semantics are `rollout_save_interval`.
      policy_dir: The directory that policies are saved in.
  """
  tf.logging.set_verbosity(tf.logging.INFO)

  policy = make_PPO2(env_name, num_vec, **make_blank_policy_kwargs)

  callback = _make_callback(
    rollout_save, rollout_save_interval, rollout_save_n_samples,
    rollout_dir, policy_save, policy_save_interval, policy_dir)

  policy.learn(total_timesteps, callback=callback)

  if rollout_save:
    save_rollouts(rollout_dir, policy, "final", rollout_save_n_samples)
  if policy_save:
    save_policies(policy_dir, policy, "final")


def _make_callback(rollout_save: bool = False,
                   rollout_save_interval: Optional[int] = None,
                   rollout_save_n_samples: Optional[int] = None,
                   rollout_dir: Optional[str] = None,

                   policy_save: bool = False,
                   policy_save_interval: Optional[int] = None,
                   policy_dir: Optional[str] = None,
                   ) -> Callable:
  """Make a callback that saves policy weights and rollouts during training.

  Arguments are the same as arguments in `main()`.
  """
  step = 0
  rollout_ok = rollout_save and rollout_save_interval is not None
  policy_ok = policy_save and policy_save_interval is not None
  os.makedirs(rollout_dir, exist_ok=True)
  os.makedirs(policy_dir, exist_ok=True)

  def callback(locals_: dict, _) -> bool:
    nonlocal step
    step += 1
    policy = locals_['self']

    if rollout_ok and step % rollout_save_interval == 0:
      save_rollouts(rollout_dir, policy, step, rollout_save_n_samples)
    if policy_ok and step % policy_save_interval == 0:
      save_policies(policy_dir, policy, step)
    return True

  return callback


def save_rollouts(rollout_dir: str,
                  policy: stable_baselines.BaseRLModel,
                  step: Union[str, int],
                  n_timesteps: int) -> None:
    """Save obs-act-obs-rew transitions from rollouts of the policy.

    Args:
        rollout_dir: Path to the save directory.
        policy: The stable baselines policy. Environment is inferred from
          `policy.get_env()`.
        step: Either the integer training step or "final" to mark that training
          is finished. Used as a suffix in the save file's basename.
        n_timesteps: The number of rollout timesteps in the save file.
    """
    filename = util.dump_prefix(policy.__class__, policy.env, step) + ".npz"
    path = osp.join(rollout_dir, filename)
    obs_old, act, obs_new, rew = util.rollout.generate_transitions(
      policy, env, n_timesteps)
    np.savez_compressed(path,
                        obs_old=obs_old, act=act, obs_new=obs_new, rew=rew)
    tf.logging.info("Dumped demonstrations to {}.".format(path))


def save_policies(policy_dir: str,
                  policy: stable_baselines.BaseRLModel,
                  step: Union[str, int]):
    """Save policy weights.

    Args:
        rollout_dir: Path to the save directory.
        policy: The stable baselines policy. Environment is inferred from
          `policy.get_env()`.
        step: Either the integer training step or "final" to mark that training
          is finished. Used as a suffix in the save file's basename.
    """
    filename = util.dump_prefix(policy.__class__, policy.env, step) + ".pkl"
    path = osp.join(policy_dir, filename)
    policy.save(path)
    tf.logging.info("Saved policy pickle to {}.".format(path))


if __name__ == "__main__":
    observer = FileStorageObserver.create(
        osp.join('output', 'sacred', 'data_collect'))
    data_collect_ex.observers.append(observer)
    data_collect_ex.run_commandline()
