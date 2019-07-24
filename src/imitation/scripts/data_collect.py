import os
import os.path as osp
from typing import Callable, Optional

from sacred.observers import FileStorageObserver
from stable_baselines import logger as sb_logger
from stable_baselines.common.vec_env import VecNormalize
import tensorflow as tf

from imitation.policies import serialize
from imitation.scripts.config.data_collect import data_collect_ex
import imitation.util as util


@data_collect_ex.main
def data_collect(_seed: int,
                 env_name: str,
                 total_timesteps: int,
                 *,
                 log_dir: str = None,
                 num_vec: int = 8,
                 parallel: bool = False,
                 normalize: bool = True,
                 make_blank_policy_kwargs: dict = {},

                 rollout_save_interval: int = 0,
                 rollout_save_final: bool = False,
                 rollout_save_n_samples: int = 2000,

                 policy_save_interval: int = -1,
                 policy_save_final: bool = True,
                 ) -> None:
  """Train a policy from scratch, optionally saving the policy and rollouts.

  At applicable training steps `step` (where step is either an integer or
  "final"):

      - Policies are saved to `{log_dir}/policies/{step}.pkl`.
      - Rollouts are saved to `{log_dir}/rollouts/{step}.pkl`.

  Args:
      env_name: The gym.Env name. Loaded as VecEnv.
      total_timesteps: Number of training timesteps in `model.learn()`.
      log_dir: The root directory to save metrics and checkpoints to.
      num_vec: Number of environments in VecEnv.
      parallel: If True, then use DummyVecEnv. Otherwise use SubprocVecEnv.
      normalize: If True, then rescale observations and reward.
      make_blank_policy_kwargs: Kwargs for `make_blank_policy`.

      rollout_save_interval: The number of training updates in between
          intermediate rollout saves. If the argument is nonpositive, then
          don't save intermediate updates.
      rollout_save_final: If True, then save rollouts right after training is
          finished.
      rollout_save_n_samples: The minimum number of timesteps saved in every
          file. Could be more than `rollout_save_n_samples` because trajectories
          are saved by episode rather than by transition.

      policy_save_interval: The number of training updates between saves. Has
          the same semantics are `rollout_save_interval`.
      policy_save_final: If True, then save the policy right after training is
          finished.
  """
  with util.make_session():
    tf.logging.set_verbosity(tf.logging.INFO)
    sb_logger.configure(folder=osp.join(log_dir, 'rl'),
                        format_strs=['tensorboard', 'stdout'])

    rollout_dir = osp.join(log_dir, "rollouts")
    policy_dir = osp.join(log_dir, "policies")
    os.makedirs(rollout_dir, exist_ok=True)
    os.makedirs(policy_dir, exist_ok=True)

    venv = util.make_vec_env(env_name, num_vec, seed=_seed,
                             parallel=parallel, log_dir=log_dir)
    vec_normalize = None
    if normalize:
      venv = vec_normalize = VecNormalize(venv)

    policy = util.init_rl(venv, verbose=1,
                          **make_blank_policy_kwargs)

    # The callback saves intermediate artifacts during training.
    callback = _make_callback(
      vec_normalize,
      rollout_save_interval, rollout_save_n_samples,
      rollout_dir, policy_save_interval, policy_dir)

    policy.learn(total_timesteps, callback=callback)

    # Save final artifacts after training is complete.
    if rollout_save_final:
      util.rollout.save(
        rollout_dir, policy, "final",
        n_timesteps=rollout_save_n_samples)
    if policy_save_final:
      output_dir = os.path.join(policy_dir, "final")
      serialize.save_stable_model(output_dir, policy, vec_normalize)


def _make_callback(vec_normalize: Optional[VecNormalize] = None,

                   rollout_save_interval: Optional[int] = None,
                   rollout_save_n_samples: Optional[int] = None,
                   rollout_dir: Optional[str] = None,

                   policy_save_interval: Optional[int] = None,
                   policy_dir: Optional[str] = None,
                   ) -> Callable:
  """Make a callback that saves policy weights and rollouts during training.

  Arguments are the same as arguments in `main()`.
  """
  step = 0
  rollout_ok = rollout_save_interval > 0
  policy_ok = policy_save_interval > 0

  def callback(locals_: dict, _) -> bool:
    nonlocal step
    step += 1
    policy = locals_['self']

    if rollout_ok and step % rollout_save_interval == 0:
      util.rollout.save(
        rollout_dir, policy, step, n_timesteps=rollout_save_n_samples)
    if policy_ok and step % policy_save_interval == 0:
      output_dir = os.path.join(policy_dir, f'{step:5d}')
      serialize.save_stable_model(output_dir, policy, vec_normalize)
    return True

  return callback


if __name__ == "__main__":
    observer = FileStorageObserver.create(
        osp.join('output', 'sacred', 'data_collect'))
    data_collect_ex.observers.append(observer)
    data_collect_ex.run_commandline()
