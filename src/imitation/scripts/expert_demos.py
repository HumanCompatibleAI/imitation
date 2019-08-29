import contextlib
import math
import os
import os.path as osp
from typing import Optional

from sacred.observers import FileStorageObserver
from stable_baselines import logger as sb_logger
from stable_baselines.common.vec_env import VecNormalize
import tensorflow as tf

import imitation.envs.examples  # noqa: F401
from imitation.policies import serialize
from imitation.rewards.serialize import load_reward
from imitation.scripts.config.expert_demos import expert_demos_ex
import imitation.util as util
from imitation.util.reward_wrapper import RewardVecEnvWrapper
from imitation.util.rollout import _validate_traj_generate_params


@expert_demos_ex.main
def rollouts_and_policy(
  _seed: int,
  env_name: str,
  total_timesteps: int,
  *,
  log_dir: str = None,
  num_vec: int = 8,
  parallel: bool = False,
  max_episode_steps: Optional[int] = None,
  normalize: bool = True,
  make_blank_policy_kwargs: dict = {},

  n_episodes_eval: int = 50,

  reward_type: Optional[str] = None,
  reward_path: Optional[str] = None,

  rollout_save_interval: int = 0,
  rollout_save_final: bool = False,
  rollout_save_n_timesteps: Optional[int] = None,
  rollout_save_n_episodes: Optional[int] = None,

  policy_save_interval: int = -1,
  policy_save_final: bool = True,
) -> dict:
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
      max_episode_steps: If not None, then environments are wrapped by
          TimeLimit so that they have at most `max_episode_steps` steps per
          episode.
      normalize: If True, then rescale observations and reward.
      make_blank_policy_kwargs: Kwargs for `make_blank_policy`.

      n_episodes_eval: The number of episodes to average over when calculating
          the average ground truth reward return of the final policy.

      reward_type: If provided, then load the serialized reward of this type,
          wrapping the environment in this reward. This is useful to test
          whether a reward model transfers. For more information, see
          `imitation.rewards.serialize.load_reward`.
      reward_path: A specifier, such as a path to a file on disk, used by
          reward_type to load the reward model. For more information, see
          `imitation.rewards.serialize.load_reward`.

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

  Returns:
      A dictionary with the following keys: "ep_reward_mean",
      "ep_reward_std_err", and "log_dir".
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
                             parallel=parallel, log_dir=log_dir,
                             max_episode_steps=max_episode_steps)

    log_callbacks = []
    with contextlib.ExitStack() as stack:
      if reward_type is not None:
        reward_fn_ctx = load_reward(reward_type, reward_path, venv)
        reward_fn = stack.enter_context(reward_fn_ctx)
        venv = RewardVecEnvWrapper(venv, reward_fn)
        log_callbacks.append(venv.log_callback)
        tf.logging.info(
            f"Wrapped env in reward {reward_type} from {reward_path}.")

      vec_normalize = None
      if normalize:
        venv = vec_normalize = VecNormalize(venv)

      policy = util.init_rl(venv, verbose=1,
                            **make_blank_policy_kwargs)

      # Make callback to save intermediate artifacts during training.
      step = 0

      def callback(locals_: dict, _) -> bool:
        nonlocal step
        step += 1
        policy = locals_['self']

        # TODO(adam): make logging frequency configurable
        for callback in log_callbacks:
          callback(sb_logger)

        if rollout_save_interval > 0 and step % rollout_save_interval == 0:
          util.rollout.save(
            rollout_dir, policy, venv, step,
            n_timesteps=rollout_save_n_timesteps,
            n_episodes=rollout_save_n_episodes)
        if policy_save_interval > 0 and step % policy_save_interval == 0:
          output_dir = os.path.join(policy_dir, f'{step:05d}')
          serialize.save_stable_model(output_dir, policy, vec_normalize)
        return True  # Continue training.

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

      # Final evaluation of expert policy.
      stats = util.rollout.rollout_stats(policy,
                                         venv,
                                         n_episodes=n_episodes_eval)
      assert stats["n_traj"] >= n_episodes_eval
      ep_reward_mean = stats["return_mean"]
      ep_reward_std_err = stats["return_std"] / math.sqrt(n_episodes_eval)
      print("[result] Mean Episode Return: "
            f"{ep_reward_mean:.4g} ± {ep_reward_std_err:.3g} "
            f"(n={stats['n_traj']})")

  return dict(ep_reward_mean=ep_reward_mean,
              ep_reward_std_err=ep_reward_std_err,
              log_dir=log_dir)


@expert_demos_ex.command
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
  rollout_save_dir: Optional[str] = None,
  max_episode_steps: Optional[int] = None,
) -> None:
  """Loads a saved policy and generates rollouts.

  Default save path is f"{log_dir}/rollouts/{env_name}.pkl". Change to
  f"{rollout_save_dir}/{env_name}.pkl" by setting the `rollout_save_dir` param.
  Unlisted arguments are the same as in `rollouts_and_policy()`.

  Args:
      policy_type: Argument to `imitation.policies.serialize.load_policy`.
      policy_path: Argument to `imitation.policies.serialize.load_policy`. If
          not provided, then defaults to f"expert_models/{env_name}".
      rollout_save_dir: Rollout pickle is saved in this directory as
          f"{env_name}.pkl".
  """
  if rollout_save_dir is None:
    rollout_save_dir = osp.join(log_dir, "rollouts")

  venv = util.make_vec_env(env_name, num_vec, seed=_seed,
                           parallel=parallel, log_dir=log_dir,
                           max_episode_steps=max_episode_steps)

  with serialize.load_policy(policy_type, policy_path, venv) as policy:
    os.makedirs(rollout_save_dir, exist_ok=True)
    util.rollout.save(rollout_save_dir, policy, venv,
                      basename=env_name,
                      n_timesteps=rollout_save_n_timesteps,
                      n_episodes=rollout_save_n_episodes,
                      )


def main_console():
  observer = FileStorageObserver.create(
      osp.join('output', 'sacred', 'expert_deoms'))
  expert_demos_ex.observers.append(observer)
  expert_demos_ex.run_commandline()


if __name__ == "__main__":
  main_console()
