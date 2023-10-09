"""Uses RL to train a policy from scratch, saving rollouts and policy.

This can be used:
    1. To train a policy on a ground-truth reward function, as a source of
       synthetic "expert" demonstrations to train IRL or imitation learning
       algorithms.
    2. To train a policy on a learned reward function, to solve a task or
       as a way of evaluating the quality of the learned reward function.
"""

import logging
import pathlib
import warnings
from typing import Any, Mapping, Optional

import numpy as np
from sacred.observers import FileStorageObserver
from stable_baselines3.common import callbacks
from stable_baselines3.common.vec_env import VecNormalize

import imitation.data.serialize as data_serialize
import imitation.policies.serialize as policies_serialize
from imitation.data import rollout, wrappers
from imitation.rewards.reward_wrapper import RewardVecEnvWrapper
from imitation.rewards.serialize import load_reward
from imitation.scripts.config.train_rl import train_rl_ex
from imitation.scripts.ingredients import environment
from imitation.scripts.ingredients import logging as logging_ingredient
from imitation.scripts.ingredients import policy_evaluation, rl


@train_rl_ex.main
def train_rl(
    *,
    total_timesteps: int,
    normalize_reward: bool,
    normalize_kwargs: dict,
    reward_type: Optional[str],
    reward_path: Optional[str],
    load_reward_kwargs: Optional[Mapping[str, Any]],
    rollout_save_final: bool,
    rollout_save_n_timesteps: Optional[int],
    rollout_save_n_episodes: Optional[int],
    policy_save_interval: int,
    policy_save_final: bool,
    agent_path: Optional[str],
    _rnd: np.random.Generator,
) -> Mapping[str, float]:
    """Trains an expert policy from scratch and saves the rollouts and policy.

    Checkpoints:
      At applicable training steps `step` (where step is either an integer or
      "final"):

        - Policies are saved to `{log_dir}/policies/{step}/`.
        - Rollouts are saved to `{log_dir}/rollouts/{step}.npz`.

    Args:
        total_timesteps: Number of training timesteps in `model.learn()`.
        normalize_reward: Applies normalization and clipping to the reward function by
            keeping a running average of training rewards. Note: this is may be
            redundant if using a learned reward that is already normalized.
        normalize_kwargs: kwargs for `VecNormalize`.
        reward_type: If provided, then load the serialized reward of this type,
            wrapping the environment in this reward. This is useful to test
            whether a reward model transfers. For more information, see
            `imitation.rewards.serialize.load_reward`.
        reward_path: A specifier, such as a path to a file on disk, used by
            reward_type to load the reward model. For more information, see
            `imitation.rewards.serialize.load_reward`.
        load_reward_kwargs: Additional kwargs to pass to `predict_processed`.
            Examples are 'alpha' for :class: `AddSTDRewardWrapper` and 'update_stats'
            for :class: `NormalizedRewardNet`.
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
        agent_path: Path to load warm-started agent.
        _rnd: Random number generator provided by Sacred.

    Returns:
        The return value of `rollout_stats()` using the final policy.
    """
    custom_logger, log_dir = logging_ingredient.setup_logging()
    rollout_dir = log_dir / "rollouts"
    policy_dir = log_dir / "policies"
    rollout_dir.mkdir(parents=True, exist_ok=True)
    policy_dir.mkdir(parents=True, exist_ok=True)

    post_wrappers = [lambda env, idx: wrappers.RolloutInfoWrapper(env)]
    with environment.make_venv(  # type: ignore[wrong-keyword-args]
        post_wrappers=post_wrappers,
    ) as venv:
        callback_objs = []
        if reward_type is not None:
            reward_fn = load_reward(
                reward_type,
                reward_path,
                venv,
                **load_reward_kwargs,
            )
            venv = RewardVecEnvWrapper(venv, reward_fn)
            callback_objs.append(venv.make_log_callback())
            logging.info(f"Wrapped env in reward {reward_type} from {reward_path}.")

        if normalize_reward:
            # Normalize reward. Reward scale effectively changes the learning rate,
            # so normalizing it makes training more stable. Note we do *not* normalize
            # observations here; use the `NormalizeFeaturesExtractor` instead.
            venv = VecNormalize(venv, norm_obs=False, **normalize_kwargs)
            if reward_type == "RewardNet_normalized":
                warnings.warn(
                    "Applying normalization to already normalized reward function. \
                    Consider setting normalize_reward as False",
                    RuntimeWarning,
                )

        if policy_save_interval > 0:
            save_policy_callback: callbacks.EventCallback = (
                policies_serialize.SavePolicyCallback(policy_dir)
            )
            save_policy_callback = callbacks.EveryNTimesteps(
                policy_save_interval,
                save_policy_callback,
            )
            callback_objs.append(save_policy_callback)
        callback = callbacks.CallbackList(callback_objs)

        if agent_path is None:
            rl_algo = rl.make_rl_algo(venv)
        else:
            rl_algo = rl.load_rl_algo_from_path(agent_path=agent_path, venv=venv)
        rl_algo.set_logger(custom_logger)
        rl_algo.learn(total_timesteps, callback=callback)

        # Save final artifacts after training is complete.
        if rollout_save_final:
            save_path = rollout_dir / "final.npz"
            sample_until = rollout.make_sample_until(
                rollout_save_n_timesteps,
                rollout_save_n_episodes,
            )
            data_serialize.save(
                save_path,
                rollout.rollout(rl_algo, rl_algo.get_env(), sample_until, rng=_rnd),
            )
        if policy_save_final:
            output_dir = policy_dir / "final"
            policies_serialize.save_stable_model(output_dir, rl_algo)

        # Final evaluation of expert policy.
        eval_stats = policy_evaluation.eval_policy(rl_algo, venv)
        return eval_stats


def main_console():
    observer_path = pathlib.Path.cwd() / "output" / "sacred" / "train_rl"
    observer = FileStorageObserver(observer_path)
    train_rl_ex.observers.append(observer)
    train_rl_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
