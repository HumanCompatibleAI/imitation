import os
import os.path as osp
from typing import Optional

from sacred.observers import FileStorageObserver
from stable_baselines import logger as sb_logger
from stable_baselines.common.vec_env import VecNormalize
import tensorflow as tf

import imitation.examples.env_suite  # noqa: F401
from imitation.policies import serialize
from imitation.scripts.config.data_collect import data_collect_ex
import imitation.util as util
from imitation.util.rollout import _validate_traj_generate_params


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
  _validate_traj_generate_params(rollout_save_n_timesteps,
                                 rollout_save_n_episodes)

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

    # Make callback to save intermediate artifacts during training.
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
        output_dir = os.path.join(policy_dir, f'{step:05d}')
        serialize.save_stable_model(output_dir, policy, vec_normalize)
      return True

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


@data_collect_ex.command
@util.make_session()
def rollouts_from_policy(
  _seed: int,
  *,
  num_vec: int,
  rollout_save_n_timesteps: int,
  rollout_save_n_episodes: int,
  log_dir: str,
  policy_path: Optional[str] = None,
  policy_type: str = "ppo2",
  env_name: str = "CartPole-v1",
  parallel: bool = True,
  rollout_save_dir: Optional[str] = None,
) -> None:
  """Loads a saved policy and generates rollouts.

  Default save path is f"{log_dir}/rollouts/{env_name}.pkl". Change to
  f"{rollout_save_dir}/{env_name}.pkl" by setting the `rollout_save_dir` param.
  Unlisted arguments are the same as in `data_collect()`.

  Args:
      policy_type: Argument to `imitation.policies.serialize.load_policy`.
      policy_path: Argument to `imitation.policies.serialize.load_policy`. If
          not provided, then defaults to f"expert_models/{env_name}".
      rollout_save_dir: Rollout pickle is saved in this directory as
          f"{env_name}.pkl".
  """
  venv = util.make_vec_env(env_name, num_vec, seed=_seed,
                           parallel=parallel, log_dir=log_dir)

  if policy_path is None:
    policy_path = f"expert_models/{env_name}"
  policy = serialize.load_policy(policy_type, policy_path, venv)

  if rollout_save_dir is None:
    rollout_save_dir = osp.join(log_dir, "rollouts")
  os.makedirs(rollout_save_dir, exist_ok=True)

  util.rollout.save(rollout_save_dir, policy, venv,
                    basename=env_name,
                    n_timesteps=rollout_save_n_timesteps,
                    n_episodes=rollout_save_n_episodes,
                    )


def main_console():
  observer = FileStorageObserver.create(
      osp.join('output', 'sacred', 'data_collect'))
  data_collect_ex.observers.append(observer)
  data_collect_ex.run_commandline()


if __name__ == "__main__":
  main_console()
