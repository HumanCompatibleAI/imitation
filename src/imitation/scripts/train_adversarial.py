"""Train GAIL or AIRL.

Can be used as a CLI script, or the `train_and_plot` function can be called directly.
"""

import logging
import os
import os.path as osp
from typing import Any, Mapping, Optional, Sequence, Tuple, Type

import torch as th
from sacred.observers import FileStorageObserver
from stable_baselines3.common import base_class, vec_env

from imitation.algorithms.adversarial import airl, common, gail
from imitation.data import rollout, types
from imitation.policies import serialize
from imitation.rewards import reward_nets
from imitation.scripts.config.train_adversarial import train_adversarial_ex
from imitation.util import logger as imit_logger
from imitation.util import sacred as sacred_util
from imitation.util import util

logger = logging.getLogger("imitation.scripts.train_adversarial")

ALGORITHMS = {
    "airl": airl.AIRL,
    "gail": gail.GAIL,
}


def save(trainer, save_path):
    """Save discriminator and generator."""
    # We implement this here and not in Trainer since we do not want to actually
    # serialize the whole Trainer (including e.g. expert demonstrations).
    os.makedirs(save_path, exist_ok=True)
    th.save(trainer.reward_train, os.path.join(save_path, "reward_train.pt"))
    th.save(trainer.reward_test, os.path.join(save_path, "reward_test.pt"))
    # TODO(gleave): unify this with the saving logic in data_collect?
    # (Needs #43 to be merged before attempting.)
    serialize.save_stable_model(
        os.path.join(save_path, "gen_policy"),
        trainer.gen_algo,
        trainer.venv_norm_obs,
    )


@train_adversarial_ex.capture
def make_reward_net(
    venv: vec_env.VecEnv,
    reward_net_cls: Optional[Type[reward_nets.RewardNet]],
    reward_net_kwargs: Optional[Mapping[str, Any]],
) -> Optional[reward_nets.RewardNet]:
    """Builds a reward network.

    Args:
        venv: Vectorized environment reward network will predict reward for.
        reward_net_cls: Class of reward network to construct.
        reward_net_kwargs: Keyword arguments passed to reward network constructor.

    Returns:
        None if `reward_net_cls` is None; otherwise, an instance of `reward_net_cls`.
    """
    if reward_net_cls is not None:
        reward_net_kwargs = reward_net_kwargs or {}
        reward_net = reward_net_cls(
            venv.observation_space,
            venv.action_space,
            **reward_net_kwargs,
        )
        logging.info(f"Reward network:\n {reward_net}")
        return reward_net


@train_adversarial_ex.capture
def get_algorithm_config(
    algorithm: str,
    algorithm_kwargs: Mapping[str, Mapping],
) -> Tuple[Type[common.AdversarialTrainer], Mapping[str, Any]]:
    """Makes an adversarial training algorithm.

    Args:
        algorithm: A case-insensitive string determining which adversarial imitation
            learning algorithm is executed. Either "airl" or "gail".
        algorithm_kwargs: Keyword arguments for the `GAIL` or `AIRL` constructor
            that can apply to either constructor. Unlike a regular kwargs argument, this
            argument can only have the following keys: "shared", "airl", and "gail".
            `algorithm_kwargs["airl"]`, if it is provided, is a kwargs `Mapping` passed
            to the `AIRL` constructor when `algorithm == "airl"`. Likewise
            `algorithm_kwargs["gail"]` is passed to the `GAIL` constructor when
            `algorithm == "gail"`. `algorithm_kwargs["shared"]`, if provided, is passed
            to both the `AIRL` and `GAIL` constructors. Duplicate keyword argument keys
            between `algorithm_kwargs["shared"]` and `algorithm_kwargs["airl"]` (or
            "gail") leads to an error.

    Returns:
        Tuple `(algo_cls, algorithm_kwargs)` where `algo_cls` is a class of an
        imitation learning algorithm and algorithm_kwargs are keyword arguments
        to pass into `algo_cls`.

    Raises:
        ValueError: `algorithm_kwargs` included unsupported key (not "shared" or
            an algorithm defined in ALGORITHMS), or `algorithm` is not defined
            in ALGORITHMS.
    """
    allowed_keys = set(ALGORITHMS.keys()).union(("shared",))
    if not algorithm_kwargs.keys() <= allowed_keys:
        raise ValueError(
            f"Invalid algorithm_kwargs.keys()={algorithm_kwargs.keys()}. "
            f"Allowed keys: {allowed_keys}",
        )

    algorithm_kwargs_shared = algorithm_kwargs.get("shared", {})
    algorithm_kwargs_algo = algorithm_kwargs.get(algorithm, {})
    final_algorithm_kwargs = dict(
        **algorithm_kwargs_shared,
        **algorithm_kwargs_algo,
    )
    try:
        algo_cls = ALGORITHMS[algorithm]
    except KeyError as e:
        raise ValueError(f"Unrecognized algorithm '{algorithm}'") from e

    logger.info(f"Using '{algorithm}' algorithm")
    return algo_cls, final_algorithm_kwargs


@train_adversarial_ex.capture
def load_expert_demos(
    rollout_path: str,
    n_expert_demos: Optional[int],
) -> Sequence[types.Trajectory]:
    """Loads expert demonstrations.

    Args:
        rollout_path: A path containing a pickled sequence of `types.Trajectory`.
        n_expert_demos: The number of trajectories to load.
            Dataset is truncated to this length if specified.

    Returns:
        The expert trajectories.

    Raises:
        ValueError: There are fewer trajectories than `n_expert_demos`.
    """
    expert_trajs = types.load(rollout_path)
    logger.info(f"Loaded {len(expert_trajs)} expert trajectories from '{rollout_path}'")
    if n_expert_demos is not None:
        if len(expert_trajs) < n_expert_demos:
            raise ValueError(
                f"Want to use n_expert_demos={n_expert_demos} trajectories, but only "
                f"{len(expert_trajs)} are available via {rollout_path}.",
            )
        expert_trajs = expert_trajs[:n_expert_demos]
        logger.info(f"Truncated to {n_expert_demos} expert trajectories")
    return expert_trajs


@train_adversarial_ex.capture
def make_rl_algo(
    venv: vec_env.VecEnv,
    gen_batch_size: int,
    rl_cls: Type[base_class.BaseAlgorithm],
    policy_cls: Type[base_class.BasePolicy],
    rl_kwargs: Mapping[str, Any],
) -> base_class.BaseAlgorithm:
    """Instantiates a Stable Baselines3 RL algorithm.

    Args:
        venv: The vectorized environment to train on.
        gen_batch_size: The batch size of the RL algorithm.
        rl_cls: Type of a Stable Baselines3 RL algorithm.
        policy_cls: Type of a Stable Baselines3 policy architecture.
        rl_kwargs: Keyword arguments for RL algorithm constructor.

    Returns:
        The RL algorithm.

    Raises:
        ValueError: `gen_batch_size` not divisible by `venv.num_envs`.
    """
    if gen_batch_size % venv.num_envs != 0:
        raise ValueError(
            f"num_envs={venv.num_envs} must evenly divide "
            f"gen_batch_size={gen_batch_size}.",
        )
    n_steps = gen_batch_size // venv.num_envs
    rl_algo = rl_cls(
        policy_cls,
        venv,
        # TODO(adam): n_steps doesn't exist in all algos -- generalize?
        n_steps=n_steps,
        **rl_kwargs,
    )
    logger.info(f"RL algorithm: {type(rl_algo)}")
    logger.info(f"Policy network summary:\n {rl_algo.policy}")
    return rl_algo


@train_adversarial_ex.capture
def eval_policy(
    rl_algo: base_class.BaseAlgorithm,
    venv: vec_env.VecEnv,
    n_episodes_eval: int,
) -> Mapping[str, float]:
    """Evaluation of imitation learned policy.

    Args:
        rl_algo: Algorithm to evaluate.
        venv: Environment to evaluate on.
        n_episodes_eval: The number of episodes to average over when calculating
            the average episode reward of the imitation policy for return.

    Returns:
        A dictionary with two keys. "imit_stats" gives the return value of
        `rollout_stats()` on rollouts test-reward-wrapped environment, using the final
        policy (remember that the ground-truth reward can be recovered from the
        "monitor_return" key). "expert_stats" gives the return value of
        `rollout_stats()` on the expert demonstrations loaded from `rollout_path`.

    """
    sample_until_eval = rollout.make_min_episodes(n_episodes_eval)
    trajs = rollout.generate_trajectories(
        rl_algo,
        venv,
        sample_until=sample_until_eval,
    )
    return rollout.rollout_stats(trajs)


@train_adversarial_ex.main
def train_adversarial(
    _run,
    _seed: int,
    env_name: str,
    env_make_kwargs: Optional[Mapping[str, Any]],
    num_vec: int,
    parallel: bool,
    max_episode_steps: Optional[int],
    log_dir: str,
    total_timesteps: int,
    checkpoint_interval: int,
) -> Mapping[str, Mapping[str, float]]:
    """Train an adversarial-network-based imitation learning algorithm.

    Checkpoints:
        - DiscrimNets are saved to `f"{log_dir}/checkpoints/{step}/discrim/"`,
            where step is either the training round or "final".
        - Generator policies are saved to `f"{log_dir}/checkpoints/{step}/gen_policy/"`.

    Args:
        _seed: Random seed.
        env_name: The environment to train in.
        env_make_kwargs: The kwargs passed to `spec.make` of a gym environment.
        num_vec: Number of `gym.Env` to vectorize.
        parallel: Whether to use "true" parallelism. If True, then use `SubProcVecEnv`.
            Otherwise, use `DummyVecEnv` which steps through environments serially.
        max_episode_steps: If not None, then a TimeLimit wrapper is applied to each
            environment to artificially limit the maximum number of timesteps in an
            episode.
        log_dir: Directory to save models and other logging to.
        total_timesteps: The number of transitions to sample from the environment
            during training.
        checkpoint_interval: Save the discriminator and generator models every
            `checkpoint_interval` rounds and after training is complete. If 0,
            then only save weights after training is complete. If <0, then don't
            save weights at all.

    Returns:
        A dictionary with two keys. "imit_stats" gives the return value of
        `rollout_stats()` on rollouts test-reward-wrapped environment, using the final
        policy (remember that the ground-truth reward can be recovered from the
        "monitor_return" key). "expert_stats" gives the return value of
        `rollout_stats()` on the expert demonstrations loaded from `rollout_path`.
    """
    custom_logger = imit_logger.configure(log_dir, ["tensorboard", "stdout"])
    os.makedirs(log_dir, exist_ok=True)
    sacred_util.build_sacred_symlink(log_dir, _run)

    venv = util.make_vec_env(
        env_name,
        num_vec,
        seed=_seed,
        parallel=parallel,
        log_dir=log_dir,
        max_episode_steps=max_episode_steps,
        env_make_kwargs=env_make_kwargs,
    )

    gen_algo = make_rl_algo(venv)
    reward_net = make_reward_net(venv)
    expert_trajs = load_expert_demos()

    algo_cls, algorithm_kwargs = get_algorithm_config()
    expert_transitions = rollout.flatten_trajectories(expert_trajs)
    logger.info(f"Loaded {len(expert_transitions)} timesteps of expert data")
    trainer = algo_cls(
        venv=venv,
        demonstrations=expert_transitions,
        gen_algo=gen_algo,
        log_dir=log_dir,
        reward_net=reward_net,
        custom_logger=custom_logger,
        **algorithm_kwargs,
    )

    def callback(round_num):
        if checkpoint_interval > 0 and round_num % checkpoint_interval == 0:
            save(trainer, os.path.join(log_dir, "checkpoints", f"{round_num:05d}"))

    trainer.train(int(total_timesteps), callback)

    # Save final artifacts.
    if checkpoint_interval >= 0:
        save(trainer, os.path.join(log_dir, "checkpoints", "final"))

    results = {}
    # TODO(adam): accessing venv_train directly is hacky
    results["imit_stats"] = eval_policy(gen_algo, trainer.venv_train)
    results["expert_stats"] = rollout.rollout_stats(expert_trajs)
    return results


def main_console():
    observer = FileStorageObserver(osp.join("output", "sacred", "train_adversarial"))
    train_adversarial_ex.observers.append(observer)
    train_adversarial_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
