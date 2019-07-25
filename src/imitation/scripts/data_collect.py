import os
import os.path as osp
from typing import Callable, Optional

from sacred.observers import FileStorageObserver
from stable_baselines import logger as sb_logger
from stable_baselines.common.vec_env import VecEnv, VecNormalize
import tensorflow as tf

from imitation.policies import serialize
from imitation.scripts.config.data_collect import data_collect_ex
import imitation.util as util


@data_collect_ex.main
def rollouts_and_policy(
  _seed: int,
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
  rollout_save_n_timesteps: Optional[int] = None,
  rollout_save_n_episodes: Optional[int] = None,

  policy_save_interval: int = -1,
  policy_save_final: bool = True,
) -> None:
  """Trains an expert policy from scratch and saves the rollouts and policy.

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
      rollout_save_n_timesteps: The minimum number of timesteps saved in every
          file. Could be more than `rollout_save_n_timesteps` because
          trajectories are saved by episode rather than by transition.
          Must set exactly one of `rollout_save_n_timesteps`
          and `rollout_save_n_episodes`.
      rollout_save_n_episodes: The number of episodes saved in every
          file. Must set exactly one of `rollout_save_n_timesteps` and
          `rollout_save_n_episodes`.
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
      venv,
      vec_normalize,
      rollout_save_interval,
      rollout_save_n_timesteps,
      rollout_save_n_episodes,
      rollout_dir, policy_save_interval, policy_dir)

    policy.learn(total_timesteps, callback=callback)

    # Save final artifacts after training is complete.
    if rollout_save_final:
      util.rollout.save(
        rollout_dir, policy, venv, "final",
        n_timesteps=rollout_save_n_timesteps,
        n_episodes=rollout_save_n_episodes)
    if policy_save_final:
      output_dir = os.path.join(policy_dir, "final")
      serialize.save_stable_model(output_dir, policy, vec_normalize)


def _make_callback(
  venv: VecEnv,
  vec_normalize: Optional[VecNormalize] = None,

  rollout_save_interval: Optional[int] = None,
  rollout_save_n_timesteps: Optional[int] = None,
  rollout_save_n_episodes: Optional[int] = None,
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
        rollout_dir, policy, venv, step,
        n_timesteps=rollout_save_n_timesteps,
        n_episodes=rollout_save_n_episodes)
    if policy_ok and step % policy_save_interval == 0:
      output_dir = os.path.join(policy_dir, f'{step:5d}')
      serialize.save_stable_model(output_dir, policy, vec_normalize)
    return True

  return callback


@data_collect_ex.command
@util.make_session()
def rollouts_from_policy(
  _seed: int,
  *,
  num_vec: int,
  rollout_save_n_timesteps: int,
  rollout_save_n_episodes: int,
  log_dir: str,
  policy_path: str,
  policy_type: str = "ppo2",
  env_name: str = "CartPole-v1",
  parallel: bool = True,
) -> None:
  """Loads a saved policy and generates rollouts.

  Save path is f"{log_dir}/rollouts/{env_name}.pkl". Unlisted arguments are the
  same as in `data_collect()`.

  Args:
      policy_type: Argument to `imitation.policies.serialize.load_policy`.
      policy_path: Argument to `imitation.policies.serialize.load_policy`.
  """
  venv = util.make_vec_env(env_name, num_vec, seed=_seed,
                           parallel=parallel, log_dir=log_dir)
  policy = serialize.load_policy(policy_type, policy_path, venv)

  rollout_dir = osp.join(log_dir, "rollouts")
  os.makedirs(rollout_dir, exist_ok=True)

  util.rollout.save(rollout_dir, policy, venv,
                    basename=env_name,
                    n_timesteps=rollout_save_n_timesteps,
                    n_episodes=rollout_save_n_episodes,
                    )


if __name__ == "__main__":
    observer = FileStorageObserver.create(
        osp.join('output', 'sacred', 'data_collect'))
    data_collect_ex.observers.append(observer)
    data_collect_ex.run_commandline()
