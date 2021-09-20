"""Train GAIL or AIRL.

Can be used as a CLI script, or the `train_and_plot` function can be called directly.
"""

import logging
import os
import os.path as osp
from typing import Any, Mapping, Optional, Tuple, Type

import torch as th
from sacred.observers import FileStorageObserver
from stable_baselines3.common import vec_env

from imitation.algorithms.adversarial import airl, gail
from imitation.algorithms.adversarial.common import AdversarialTrainer
from imitation.data import rollout
from imitation.policies import serialize
from imitation.rewards import reward_nets
from imitation.scripts.common import rl, train
from imitation.scripts.config.train_adversarial import train_adversarial_ex

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
) -> Tuple[Type[AdversarialTrainer], Mapping[str, Any]]:
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
    try:
        algo_cls = ALGORITHMS[algorithm]
    except KeyError as e:
        raise ValueError(f"Unrecognized algorithm '{algorithm}'") from e

    allowed_keys = set(ALGORITHMS.keys()).union(("shared",))
    if not algorithm_kwargs.keys() <= allowed_keys:
        raise ValueError(
            f"Invalid algorithm_kwargs.keys()={algorithm_kwargs.keys()}. "
            f"Allowed keys: {allowed_keys}",
        )

    algorithm_kwargs_shared = dict(algorithm_kwargs.get("shared", {}))
    algorithm_kwargs_algo = algorithm_kwargs.get(algorithm, {})
    final_algorithm_kwargs = dict(
        **algorithm_kwargs_shared,
        **algorithm_kwargs_algo,
    )

    logger.info(f"Using '{algorithm}' algorithm")
    return algo_cls, final_algorithm_kwargs


@train_adversarial_ex.main
def train_adversarial(
    _run,
    _seed: int,
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
        `rollout_stats()` on the expert demonstrations.
    """
    custom_logger, log_dir = train.setup_logging()

    expert_trajs = train.load_expert_demos()
    expert_transitions = rollout.flatten_trajectories(expert_trajs)
    logger.info(f"Loaded {len(expert_transitions)} timesteps of expert data")

    venv = train.make_venv()
    gen_algo = rl.make_rl_algo(venv)
    reward_net = make_reward_net(venv)

    algo_cls, algorithm_kwargs = get_algorithm_config()
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
    results["imit_stats"] = train.eval_policy(trainer.policy, trainer.venv_train)
    results["expert_stats"] = rollout.rollout_stats(expert_trajs)
    return results


def main_console():
    observer = FileStorageObserver(osp.join("output", "sacred", "train_adversarial"))
    train_adversarial_ex.observers.append(observer)
    train_adversarial_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
