"""Uses RL to train a policy from scratch, saving rollouts and policy.

This can be used:
    1. To train a policy on a ground-truth reward function, as a source of
       synthetic "expert" demonstrations to train IRL or imitation learning
       algorithms.
    2. To train a policy on a learned reward function, to solve a task or
       as a way of evaluating the quality of the learned reward function.
"""

import logging
import os
import os.path as osp
from typing import Mapping, Optional

import sacred.run
from sacred.observers import FileStorageObserver
from stable_baselines3.common import callbacks
from stable_baselines3.common.vec_env import VecNormalize

from imitation.data import rollout, wrappers
from imitation.policies import serialize
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.rewards.serialize import load_reward
from imitation.scripts.common import common, rl, train
from imitation.scripts.config.train_rl import train_rl_ex


@train_rl_ex.main
def train_rl(
    *,
    _run: sacred.run.Run,
    _seed: int,
    total_timesteps: int,
    normalize: bool,
    normalize_kwargs: dict,
    reward_type: Optional[str],
    reward_path: Optional[str],
    rollout_save_final: bool,
    rollout_save_n_timesteps: Optional[int],
    rollout_save_n_episodes: Optional[int],
    policy_save_interval: int,
    policy_save_final: bool,
) -> Mapping[str, float]:
    """Trains an expert policy from scratch and saves the rollouts and policy.

    Checkpoints:
      At applicable training steps `step` (where step is either an integer or
      "final"):

        - Policies are saved to `{log_dir}/policies/{step}/`.
        - Rollouts are saved to `{log_dir}/rollouts/{step}.pkl`.

    Args:
        total_timesteps: Number of training timesteps in `model.learn()`.
        normalize: If True, then rescale observations and reward.
        normalize_kwargs: kwargs for `VecNormalize`.
        reward_type: If provided, then load the serialized reward of this type,
            wrapping the environment in this reward. This is useful to test
            whether a reward model transfers. For more information, see
            `imitation.rewards.serialize.load_reward`.
        reward_path: A specifier, such as a path to a file on disk, used by
            reward_type to load the reward model. For more information, see
            `imitation.rewards.serialize.load_reward`.
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
        policy_save_interval: The number of training updates between in between
            intermediate rollout saves. If the argument is nonpositive, then
            don't save intermediate updates.
        policy_save_final: If True, then save the policy right after training is
            finished.

    Returns:
        The return value of `rollout_stats()` using the final policy.
    """
    custom_logger, log_dir = common.setup_logging()
    rollout_dir = osp.join(log_dir, "rollouts")
    policy_dir = osp.join(log_dir, "policies")
    os.makedirs(rollout_dir, exist_ok=True)
    os.makedirs(policy_dir, exist_ok=True)

    venv = common.make_venv(
        post_wrappers=[lambda env, idx: wrappers.RolloutInfoWrapper(env)],
    )
    callback_objs = []

    if reward_type is not None:
        reward_fn = load_reward(reward_type, reward_path, venv)
        venv = RewardVecEnvWrapper(venv, reward_fn)
        callback_objs.append(venv.make_log_callback())
        logging.info(f"Wrapped env in reward {reward_type} from {reward_path}.")

    vec_normalize = None
    if normalize:
        venv = vec_normalize = VecNormalize(venv, **normalize_kwargs)

    if policy_save_interval > 0:
        save_policy_callback = serialize.SavePolicyCallback(policy_dir, vec_normalize)
        save_policy_callback = callbacks.EveryNTimesteps(
            policy_save_interval,
            save_policy_callback,
        )
        callback_objs.append(save_policy_callback)
    callback = callbacks.CallbackList(callback_objs)

    rl_algo = rl.make_rl_algo(venv)
    rl_algo.set_logger(custom_logger)
    rl_algo.learn(total_timesteps, callback=callback)

    # Save final artifacts after training is complete.
    if rollout_save_final:
        save_path = osp.join(rollout_dir, "final.pkl")
        sample_until = rollout.make_sample_until(
            rollout_save_n_timesteps,
            rollout_save_n_episodes,
        )
        rollout.rollout_and_save(save_path, rl_algo, venv, sample_until)
    if policy_save_final:
        output_dir = os.path.join(policy_dir, "final")
        serialize.save_stable_model(output_dir, rl_algo, vec_normalize)

    # Final evaluation of expert policy.
    return train.eval_policy(rl_algo, venv)


def main_console():
    observer = FileStorageObserver(osp.join("output", "sacred", "train_rl"))
    train_rl_ex.observers.append(observer)
    train_rl_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
