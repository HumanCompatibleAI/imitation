"""Train GAIL or AIRL.

Can be used as a CLI script, or the `train_and_plot` function can be called directly.
"""

import os
import os.path as osp
from typing import Mapping, Optional

import tensorflow as tf
from sacred.observers import FileStorageObserver

from imitation.algorithms import adversarial
from imitation.data import rollout, types
from imitation.policies import serialize
from imitation.scripts.config.train_adversarial import train_ex
from imitation.util import logger, networks
from imitation.util import sacred as sacred_util
from imitation.util import util


def save(trainer, save_path):
    """Save discriminator and generator."""
    # We implement this here and not in Trainer since we do not want to actually
    # serialize the whole Trainer (including e.g. expert demonstrations).
    trainer.discrim.save(os.path.join(save_path, "discrim"))
    # TODO(gleave): unify this with the saving logic in data_collect?
    # (Needs #43 to be merged before attempting.)
    serialize.save_stable_model(
        os.path.join(save_path, "gen_policy"),
        trainer.gen_policy,
        trainer.venv_norm_obs,
    )


@train_ex.main
def train(
    _run,
    _seed: int,
    algorithm: str,
    env_name: str,
    num_vec: int,
    parallel: bool,
    max_episode_steps: Optional[int],
    rollout_path: str,
    n_expert_demos: Optional[int],
    log_dir: str,
    total_timesteps: int,
    n_episodes_eval: int,
    init_tensorboard: bool,
    checkpoint_interval: int,
    init_rl_kwargs: Mapping,
    algorithm_kwargs: Mapping[str, Mapping],
    discrim_net_kwargs: Mapping[str, Mapping],
) -> dict:
    """Train an adversarial-network-based imitation learning algorithm.

    Checkpoints:
        - DiscrimNets are saved to `f"{log_dir}/checkpoints/{step}/discrim/"`,
            where step is either the training epoch or "final".
        - Generator policies are saved to `f"{log_dir}/checkpoints/{step}/gen_policy/"`.

    Args:
        _seed: Random seed.
        algorithm: A case-insensitive string determining which adversarial imitation
            learning algorithm is executed. Either "airl" or "gail".
        env_name: The environment to train in.
        num_vec: Number of `gym.Env` to vectorize.
        parallel: Whether to use "true" parallelism. If True, then use `SubProcVecEnv`.
            Otherwise, use `DummyVecEnv` which steps through environments serially.
        max_episode_steps: If not None, then a TimeLimit wrapper is applied to each
            environment to artificially limit the maximum number of timesteps in an
            episode.
        rollout_path: Path to pickle containing list of Trajectories. Used as
            expert demonstrations.
        n_expert_demos: The number of expert trajectories to actually use
            after loading them from `rollout_path`.
            If None, then use all available trajectories.
            If `n_expert_demos` is an `int`, then use exactly `n_expert_demos`
            trajectories, erroring if there aren't enough trajectories. If there are
            surplus trajectories, then use the first `n_expert_demos` trajectories and
            drop the rest.
        log_dir: Directory to save models and other logging to.
        total_timesteps: The number of transitions to sample from the environment
            during training.
        n_episodes_eval: The number of episodes to average over when calculating
            the average episode reward of the imitation policy for return.
        init_tensorboard: If True, then write tensorboard logs to `{log_dir}/sb_tb`.
        checkpoint_interval: Save the discriminator and generator models every
            `checkpoint_interval` epochs and after training is complete. If 0,
            then only save weights after training is complete. If <0, then don't
            save weights at all.
        init_rl_kwargs: Keyword arguments for `init_rl`, the RL algorithm initialization
            utility function.
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
        discrim_net_kwargs: Keyword arguments for the `DiscrimNet` constructor. Unlike a
            regular kwargs argument, this argument can only have the following keys:
            "shared", "airl", "gail". These keys have the same meaning as they do in
            `algorithm_kwargs`.

    Returns:
        A dictionary with two keys. "imit_stats" gives the return value of
        `rollout_stats()` on rollouts test-reward-wrapped environment, using the final
        policy (remember that the ground-truth reward can be recovered from the
        "monitor_return" key). "expert_stats" gives the return value of
        `rollout_stats()` on the expert demonstrations loaded from `rollout_path`.
    """
    assert os.path.exists(rollout_path)
    total_timesteps = int(total_timesteps)

    tf.logging.info("Logging to %s", log_dir)
    logger.configure(log_dir, ["tensorboard", "stdout"])
    os.makedirs(log_dir, exist_ok=True)
    sacred_util.build_sacred_symlink(log_dir, _run)

    expert_trajs = types.load(rollout_path)
    if n_expert_demos is not None:
        assert len(expert_trajs) >= n_expert_demos
        expert_trajs = expert_trajs[:n_expert_demos]
    expert_transitions = rollout.flatten_trajectories(expert_trajs)

    with networks.make_session():
        if init_tensorboard:
            tensorboard_log = osp.join(log_dir, "sb_tb")
        else:
            tensorboard_log = None

        venv = util.make_vec_env(
            env_name,
            num_vec,
            seed=_seed,
            parallel=parallel,
            log_dir=log_dir,
            max_episode_steps=max_episode_steps,
        )

        # TODO(shwang): Let's get rid of init_rl later on?
        # It's really just a stub function now.
        gen_policy = util.init_rl(
            venv, verbose=1, tensorboard_log=tensorboard_log, **init_rl_kwargs
        )

        # Convert Sacred's ReadOnlyDict to dict so we can modify it.
        allowed_keys = {"shared", "gail", "airl"}
        assert discrim_net_kwargs.keys() <= allowed_keys
        assert algorithm_kwargs.keys() <= allowed_keys

        discrim_kwargs_shared = discrim_net_kwargs.get("shared", {})
        discrim_kwargs_algo = discrim_net_kwargs.get(algorithm, {})
        final_discrim_kwargs = dict(**discrim_kwargs_shared, **discrim_kwargs_algo)

        algorithm_kwargs_shared = algorithm_kwargs.get("shared", {})
        algorithm_kwargs_algo = algorithm_kwargs.get(algorithm, {})
        final_algorithm_kwargs = dict(
            **algorithm_kwargs_shared,
            **algorithm_kwargs_algo,
        )

        if algorithm.lower() == "gail":
            algo_cls = adversarial.GAIL
        elif algorithm.lower() == "airl":
            algo_cls = adversarial.AIRL
        else:
            raise ValueError(f"Invalid value algorithm={algorithm}.")

        trainer = algo_cls(
            venv=venv,
            expert_data=expert_transitions,
            gen_policy=gen_policy,
            log_dir=log_dir,
            discrim_kwargs=final_discrim_kwargs,
            **final_algorithm_kwargs,
        )

        def callback(epoch):
            if checkpoint_interval > 0 and epoch % checkpoint_interval == 0:
                save(trainer, os.path.join(log_dir, "checkpoints", f"{epoch:05d}"))

        trainer.train(total_timesteps, callback)

        # Save final artifacts.
        if checkpoint_interval >= 0:
            save(trainer, os.path.join(log_dir, "checkpoints", "final"))

        # Final evaluation of imitation policy.
        results = {}
        sample_until_eval = rollout.min_episodes(n_episodes_eval)
        trajs = rollout.generate_trajectories(
            trainer.gen_policy, trainer.venv_train, sample_until=sample_until_eval
        )
        results["expert_stats"] = rollout.rollout_stats(expert_trajs)
        results["imit_stats"] = rollout.rollout_stats(trajs)
        return results


def main_console():
    observer = FileStorageObserver(osp.join("output", "sacred", "train"))
    train_ex.observers.append(observer)
    train_ex.run_commandline()


if __name__ == "__main__":  # pragma: no cover
    main_console()
